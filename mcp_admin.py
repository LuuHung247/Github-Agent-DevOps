"""
GitHub User Management MCP - Security Hardened Version
Addresses: Privilege Escalation, RBAC, Rate Limit DoS, Performance
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import httpx
import os
import asyncio
import logging
import hashlib
import time
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

# ============= Initialize MCP =============
mcp = FastMCP("github-admin")

# ============= Security Configuration =============
@dataclass
class SecurityConfig:
    """Security policies cho MCP server"""
    
    # RBAC: Allowed usernames có thể gọi admin commands
    allowed_admins: Set[str] = field(default_factory=lambda: {
        admin.strip() for admin in os.environ.get("MCP_ALLOWED_ADMINS", "").split(",") if admin.strip()
    })
    
    # Maximum permissions mà MCP được phép cấp
    max_permission_level: str = os.environ.get("MCP_MAX_PERMISSION", "push")
    
    # Whitelist repos được phép quản lý
    allowed_repo_patterns: List[str] = field(default_factory=lambda: 
        os.environ.get("MCP_ALLOWED_REPOS", "*/*").split(",")
    )
    
    # Rate limiting per caller
    max_requests_per_minute: int = int(os.environ.get("MCP_RATE_LIMIT", "30"))
    
    # Audit log
    audit_log_path: str = os.environ.get("MCP_AUDIT_LOG", "./mcp_audit.log")
    
    # Require approval cho sensitive operations
    require_approval_for_admin: bool = os.environ.get("MCP_REQUIRE_APPROVAL", "true").lower() == "true"


security_config = SecurityConfig()

# ============= Error Codes =============
class ErrorCode(Enum):
    """Error codes cho structured error handling"""
    PERMISSION_DENIED = "PERMISSION_DENIED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    NOT_FOUND = "NOT_FOUND"
    API_ERROR = "API_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"


# ============= Tool Result =============
@dataclass
class ToolResult:
    """Structured result cho MCP tools"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_response(self) -> List[TextContent]:
        """Convert to MCP TextContent response"""
        if self.success:
            return [TextContent(
                type="text",
                text=json.dumps(self.data, indent=2, ensure_ascii=False)
            )]
        else:
            error_obj = {
                "error": True,
                "code": self.error_code,
                "message": self.error_message
            }
            return [TextContent(
                type="text",
                text=json.dumps(error_obj, indent=2, ensure_ascii=False)
            )]


# ============= GitHub Client =============
class GitHubClient:
    """GitHub API client với caching và error handling"""
    
    def __init__(self):
        self.token = os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable not set")
        
        self.base_url = "https://api.github.com"
        self._cache: Dict[str, tuple] = {}  # (data, timestamp)
        self._cache_ttl = 300  # 5 minutes
        
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def _get_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate cache key"""
        params_str = json.dumps(params or {}, sort_keys=True)
        return hashlib.md5(f"{endpoint}:{params_str}".encode()).hexdigest()
    
    def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries matching pattern"""
        if pattern is None:
            self._cache.clear()
        else:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
    
    async def request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        json_data: Dict = None,
        use_cache: bool = True
    ) -> ToolResult:
        """Make GitHub API request"""
        
        # Check cache for GET requests
        if method == "GET" and use_cache:
            cache_key = self._get_cache_key(endpoint, params)
            if cache_key in self._cache:
                data, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return ToolResult(success=True, data=data)
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=self._get_headers(),
                    params=params,
                    json=json_data
                )
                
                if response.status_code == 404:
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.NOT_FOUND.value,
                        error_message=f"Resource not found: {endpoint}"
                    )
                
                if response.status_code >= 400:
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.API_ERROR.value,
                        error_message=f"API error {response.status_code}: {response.text}"
                    )
                
                data = response.json() if response.text else {}
                
                # Cache GET requests
                if method == "GET" and use_cache:
                    cache_key = self._get_cache_key(endpoint, params)
                    self._cache[cache_key] = (data, time.time())
                
                return ToolResult(success=True, data=data)
                
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error_code=ErrorCode.NETWORK_ERROR.value,
                error_message="Request timeout"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_code=ErrorCode.NETWORK_ERROR.value,
                error_message=f"Network error: {str(e)}"
            )


# Global client instance
github_client = GitHubClient()


# ============= Audit Logger =============
audit_logger = logging.getLogger("mcp_audit")
audit_handler = logging.FileHandler(security_config.audit_log_path)
audit_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'
))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)

def audit_log(action: str, details: Dict, caller: str = "unknown", success: bool = True):
    """Log mọi action quan trọng"""
    audit_logger.info(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "caller": caller,
        "action": action,
        "success": success,
        "details": details
    }))


# ============= RBAC Enforcement =============
class PermissionLevel(Enum):
    """Hierarchy của permissions"""
    PULL = 1
    TRIAGE = 2
    PUSH = 3
    MAINTAIN = 4
    ADMIN = 5


def get_permission_level(permission: str) -> int:
    """Convert permission string to numeric level"""
    try:
        return PermissionLevel[permission.upper()].value
    except KeyError:
        return 0


def check_permission_allowed(requested_permission: str) -> bool:
    """Kiểm tra permission có vượt quá max_permission_level không"""
    max_level = get_permission_level(security_config.max_permission_level)
    requested_level = get_permission_level(requested_permission)
    return requested_level <= max_level


def check_repo_allowed(owner: str, repo: str) -> bool:
    """Kiểm tra repo có trong whitelist không"""
    full_name = f"{owner}/{repo}"
    
    for pattern in security_config.allowed_repo_patterns:
        if pattern == "*/*":  # Allow all
            return True
        
        # Simple wildcard matching
        pattern_regex = pattern.replace("*", ".*")
        if re.match(f"^{pattern_regex}$", full_name):
            return True
    
    return False


def require_client(func):
    """Decorator để inject GitHub client"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper


def require_permission(min_permission: str = "pull"):
    """Decorator kiểm tra caller có đủ quyền không"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get caller info from MCP context (if available)
            caller = kwargs.get("_mcp_caller", "unknown")
            
            # Check if caller is in allowed_admins
            if security_config.allowed_admins and caller not in security_config.allowed_admins:
                audit_log(
                    action=func.__name__,
                    details={"reason": "unauthorized_caller", "args": str(args)},
                    caller=caller,
                    success=False
                )
                return ToolResult(
                    success=False,
                    error_code=ErrorCode.PERMISSION_DENIED.value,
                    error_message=f"Caller '{caller}' not authorized for this operation"
                ).to_response()
            
            # Check permission level
            if "permission" in kwargs:
                requested = kwargs["permission"]
                if not check_permission_allowed(requested):
                    audit_log(
                        action=func.__name__,
                        details={"reason": "permission_too_high", "requested": requested},
                        caller=caller,
                        success=False
                    )
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.PERMISSION_DENIED.value,
                        error_message=f"Cannot grant '{requested}' permission. Max allowed: '{security_config.max_permission_level}'"
                    ).to_response()
            
            # Check repo whitelist
            if "owner" in kwargs and "repo" in kwargs:
                if not check_repo_allowed(kwargs["owner"], kwargs["repo"]):
                    audit_log(
                        action=func.__name__,
                        details={"reason": "repo_not_whitelisted", "repo": f"{kwargs['owner']}/{kwargs['repo']}"},
                        caller=caller,
                        success=False
                    )
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.PERMISSION_DENIED.value,
                        error_message=f"Repository '{kwargs['owner']}/{kwargs['repo']}' not in allowed list"
                    ).to_response()
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Audit log success
            audit_log(
                action=func.__name__,
                details={"args": {k: v for k, v in kwargs.items() if not k.startswith("_")}},
                caller=caller,
                success=True
            )
            
            return result
        
        return wrapper
    return decorator


# ============= Per-Caller Rate Limiting =============
class CallerRateLimiter:
    """Rate limiter per caller để tránh DoS"""
    
    def __init__(self):
        self._callers: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self, caller: str) -> bool:
        """Check nếu caller vượt quá rate limit"""
        async with self._lock:
            now = time.time()
            window_start = now - 60  # 1 minute window
            
            # Get requests trong 1 phút qua
            if caller not in self._callers:
                self._callers[caller] = []
            
            # Clean old requests
            self._callers[caller] = [
                t for t in self._callers[caller] 
                if t > window_start
            ]
            
            # Check limit
            if len(self._callers[caller]) >= security_config.max_requests_per_minute:
                return False
            
            # Add current request
            self._callers[caller].append(now)
            return True


caller_rate_limiter = CallerRateLimiter()


def rate_limit_caller(func):
    """Decorator kiểm tra rate limit per caller"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        caller = kwargs.get("_mcp_caller", "unknown")
        
        if not await caller_rate_limiter.check_rate_limit(caller):
            audit_log(
                action=func.__name__,
                details={"reason": "rate_limit_exceeded"},
                caller=caller,
                success=False
            )
            return ToolResult(
                success=False,
                error_code=ErrorCode.RATE_LIMITED.value,
                error_message=f"Rate limit exceeded: {security_config.max_requests_per_minute} requests/minute"
            ).to_response()
        
        return await func(*args, **kwargs)
    
    return wrapper


# ============= Validation Functions =============
def validate_repo_name(name: str) -> bool:
    if not name or len(name) > 100:
        return False
    return bool(re.match(r'^[a-zA-Z0-9._-]+$', name))

def validate_username(username: str) -> bool:
    if not username or len(username) > 39:
        return False
    return bool(re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$', username))

def validate_permission(permission: str) -> bool:
    valid_permissions = {"pull", "push", "admin", "maintain", "triage"}
    return permission.lower() in valid_permissions


# ============= MCP Tools =============

@mcp.tool()
@require_client
async def list_repos(
    owner: str,
    type: str = "owner",
    _mcp_caller: str = "unknown"
) -> List[TextContent]:
    """
    List repositories for a user or organization
    
    Args:
        owner: GitHub username or organization
        type: "owner", "member", or "all"
    """
    result = await github_client.request(
        "GET",
        f"/users/{owner}/repos",
        params={"type": type, "per_page": 100}
    )
    
    if not result.success:
        return result.to_response()
    
    repos = [{
        "full_name": r["full_name"],
        "owner": r["owner"]["login"],
        "name": r["name"],
        "private": r["private"],
        "description": r.get("description", "")
    } for r in result.data]
    
    return ToolResult(
        success=True,
        data={"total": len(repos), "repositories": repos}
    ).to_response()


@mcp.tool()
@require_client
async def list_collaborators(
    owner: str,
    repo: str,
    _mcp_caller: str = "unknown"
) -> List[TextContent]:
    """
    List collaborators for a repository
    
    Args:
        owner: Repository owner
        repo: Repository name
    """
    if not validate_username(owner) or not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid owner or repo name"
        ).to_response()
    
    result = await github_client.request(
        "GET",
        f"/repos/{owner}/{repo}/collaborators",
        params={"per_page": 100}
    )
    
    if not result.success:
        return result.to_response()
    
    collaborators = [{
        "login": c["login"],
        "permissions": c.get("permissions", {}),
        "role_name": c.get("role_name", "unknown")
    } for c in result.data]
    
    return ToolResult(
        success=True,
        data={
            "repository": f"{owner}/{repo}",
            "total": len(collaborators),
            "collaborators": collaborators
        }
    ).to_response()


@mcp.tool()
@require_client
@rate_limit_caller
@require_permission("admin")
async def add_collaborator(
    owner: str,
    repo: str,
    username: str,
    permission: str = "pull",
    _mcp_caller: str = "unknown"
) -> List[TextContent]:
    """
    Add collaborator với RBAC và audit logging.
    
    Args:
        owner: Repository owner
        repo: Repository name
        username: Username to add
        permission: Permission level (pull, push, maintain, triage, admin)
    
    Security:
    - Checks caller authorization
    - Validates permission level
    - Checks repo whitelist
    - Logs all actions
    """
    
    # Validation
    if not validate_username(owner) or not validate_username(username):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid username"
        ).to_response()
    
    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repo name"
        ).to_response()
    
    if not validate_permission(permission):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Invalid permission: {permission}"
        ).to_response()
    
    # Admin permission requires approval
    if permission == "admin" and security_config.require_approval_for_admin:
        return ToolResult(
            success=False,
            error_code=ErrorCode.PERMISSION_DENIED.value,
            error_message="Admin permission requires manual approval. Use GitHub UI."
        ).to_response()
    
    # Execute
    result = await github_client.request(
        "PUT", f"/repos/{owner}/{repo}/collaborators/{username}",
        json_data={"permission": permission.lower()}
    )
    
    if not result.success:
        return result.to_response()
    
    github_client.invalidate_cache(f"{owner}/{repo}")
    
    return ToolResult(
        success=True,
        data={
            "action": "invitation_sent",
            "username": username,
            "repository": f"{owner}/{repo}",
            "permission": permission,
            "caller": _mcp_caller
        }
    ).to_response()


# ============= Security Monitoring Tools =============

@mcp.tool()
async def security_status(_mcp_caller: str = "unknown") -> List[TextContent]:
    """
    Xem security configuration hiện tại
    """
    return ToolResult(
        success=True,
        data={
            "rbac_enabled": bool(security_config.allowed_admins),
            "allowed_admins": list(security_config.allowed_admins) if security_config.allowed_admins else ["*"],
            "max_permission_level": security_config.max_permission_level,
            "repo_whitelist": security_config.allowed_repo_patterns,
            "rate_limit": f"{security_config.max_requests_per_minute} requests/minute",
            "admin_approval_required": security_config.require_approval_for_admin,
            "audit_log": security_config.audit_log_path
        }
    ).to_response()


@mcp.tool()
async def audit_log_recent(limit: int = 20, _mcp_caller: str = "unknown") -> List[TextContent]:
    """
    Xem audit logs gần đây
    """
    try:
        with open(security_config.audit_log_path, "r") as f:
            lines = f.readlines()
            recent = lines[-limit:] if len(lines) > limit else lines
            logs = [json.loads(line.split(" | ")[-1]) for line in recent if line.strip()]
            
        return ToolResult(
            success=True,
            data={"total": len(logs), "logs": logs}
        ).to_response()
    except FileNotFoundError:
        return ToolResult(
            success=True,
            data={"total": 0, "logs": [], "message": "No audit log file yet"}
        ).to_response()


# ============= Main =============
if __name__ == "__main__":
    print("🔒 GitHub User Management MCP - Security Hardened")
    print("=" * 60)
    print(f"✓ RBAC: {'Enabled' if security_config.allowed_admins else 'Disabled'}")
    print(f"✓ Max Permission: {security_config.max_permission_level}")
    print(f"✓ Rate Limit: {security_config.max_requests_per_minute}/min per caller")
    print(f"✓ Audit Log: {security_config.audit_log_path}")
    print("=" * 60)
    
    mcp.run()