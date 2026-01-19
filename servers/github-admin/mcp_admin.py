"""
GitHub User Management MCP Server

Manages repository collaborators via GitHub API.
Supports both personal accounts and organizations with appropriate permission models.

Reference: https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-user-account-settings/permission-levels-for-a-personal-account-repository
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
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

mcp = FastMCP("github-admin")


@dataclass
class SecurityConfig:
    """
    Security configuration for MCP server.

    Security layers:
    - Input validation: Prevents injection attacks
    - Repo whitelist: Restricts operations to specific repositories
    - Rate limiting: Prevents abuse
    - Audit logging: Records all operations
    - GitHub token scope: Enforced by GitHub API
    """

    allowed_repo_patterns: List[str] = field(default_factory=lambda:
        [p.strip() for p in os.environ.get("MCP_ALLOWED_REPOS", "*/*").split(",") if p.strip()]
    )
    max_requests_per_minute: int = int(os.environ.get("MCP_RATE_LIMIT", "30"))
    audit_log_path: str = os.environ.get("MCP_AUDIT_LOG", "./logs/mcp_audit.log")


config = SecurityConfig()


class ErrorCode(Enum):
    """Standardized error codes for API responses."""
    PERMISSION_DENIED = "PERMISSION_DENIED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    NOT_FOUND = "NOT_FOUND"
    API_ERROR = "API_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    TOKEN_ERROR = "TOKEN_ERROR"


# =============================================================================
# Result Handling
# =============================================================================

@dataclass
class ToolResult:
    """Structured result for MCP tool responses."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_response(self) -> List[TextContent]:
        if self.success:
            return [TextContent(type="text", text=json.dumps(self.data, indent=2, ensure_ascii=False))]
        return [TextContent(type="text", text=json.dumps({
            "error": True,
            "code": self.error_code,
            "message": self.error_message
        }, indent=2, ensure_ascii=False))]


# =============================================================================
# GitHub API Client
# =============================================================================

class GitHubClient:
    """
    GitHub API client with caching and automatic owner detection.

    Handles both personal accounts and organizations transparently.
    """

    def __init__(self):
        self.token = os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable is required")

        self.base_url = "https://api.github.com"
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 300
        self._token_owner: Optional[str] = None
        self._token_scopes: Optional[List[str]] = None
        self._owner_type_cache: Dict[str, str] = {}

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

    async def get_token_info(self) -> ToolResult:
        """Retrieve token owner information and scopes."""
        if self._token_owner:
            return ToolResult(success=True, data={
                "owner": self._token_owner,
                "scopes": self._token_scopes
            })

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/user", headers=self._headers())

                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.TOKEN_ERROR.value,
                        error_message=f"Token validation failed: HTTP {response.status_code}"
                    )

                data = response.json()
                self._token_owner = data["login"]
                scopes_header = response.headers.get("x-oauth-scopes", "")
                self._token_scopes = [s.strip() for s in scopes_header.split(",") if s.strip()]

                return ToolResult(success=True, data={
                    "owner": self._token_owner,
                    "name": data.get("name", ""),
                    "email": data.get("email", ""),
                    "scopes": self._token_scopes,
                    "public_repos": data.get("public_repos", 0),
                    "private_repos": data.get("total_private_repos", 0)
                })
        except Exception as e:
            return ToolResult(
                success=False,
                error_code=ErrorCode.NETWORK_ERROR.value,
                error_message=f"Network error: {str(e)}"
            )

    async def get_owner_type(self, owner: str) -> str:
        """
        Determine if owner is a User or Organization.

        Returns: "User" or "Organization"
        """
        if owner in self._owner_type_cache:
            return self._owner_type_cache[owner]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/users/{owner}", headers=self._headers())
                if response.status_code == 200:
                    owner_type = response.json().get("type", "User")
                    self._owner_type_cache[owner] = owner_type
                    return owner_type
        except:
            pass

        return "User"

    def _cache_key(self, endpoint: str, params: Dict = None) -> str:
        params_str = json.dumps(params or {}, sort_keys=True)
        return hashlib.md5(f"{endpoint}:{params_str}".encode()).hexdigest()

    def invalidate_cache(self, pattern: str = None):
        if pattern is None:
            self._cache.clear()
        else:
            self._cache = {k: v for k, v in self._cache.items() if pattern not in k}

    async def request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        json_data: Dict = None,
        use_cache: bool = True
    ) -> ToolResult:
        """Execute GitHub API request with caching support."""

        if method == "GET" and use_cache:
            key = self._cache_key(endpoint, params)
            if key in self._cache:
                data, timestamp = self._cache[key]
                if time.time() - timestamp < self._cache_ttl:
                    return ToolResult(success=True, data=data)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method=method,
                    url=f"{self.base_url}{endpoint}",
                    headers=self._headers(),
                    params=params,
                    json=json_data
                )

                if response.status_code == 404:
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.NOT_FOUND.value,
                        error_message=f"Resource not found: {endpoint}"
                    )

                if response.status_code == 403:
                    error_msg = response.json().get("message", "Access denied")
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.PERMISSION_DENIED.value,
                        error_message=f"Permission denied: {error_msg}"
                    )

                if response.status_code >= 400:
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.API_ERROR.value,
                        error_message=f"API error {response.status_code}: {response.text}"
                    )

                data = response.json() if response.text else {}

                if method == "GET" and use_cache:
                    self._cache[self._cache_key(endpoint, params)] = (data, time.time())

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


github = GitHubClient()


# =============================================================================
# Audit Logging
# =============================================================================

audit_logger = logging.getLogger("mcp_audit")
_handler = logging.FileHandler(config.audit_log_path)
_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
audit_logger.addHandler(_handler)
audit_logger.setLevel(logging.INFO)


def audit_log(action: str, details: Dict, caller: str = "unknown", success: bool = True):
    """Record operation to audit log."""
    audit_logger.info(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "caller": caller,
        "action": action,
        "success": success,
        "details": details
    }))


# =============================================================================
# Validation
# =============================================================================

def validate_repo_name(name: str) -> bool:
    if not name or len(name) > 100:
        return False
    return bool(re.match(r'^[a-zA-Z0-9._-]+$', name))


def validate_username(username: str) -> bool:
    if not username or len(username) > 39:
        return False
    return bool(re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$', username))


def check_repo_allowed(owner: str, repo: str) -> bool:
    """Check if repository is in whitelist."""
    full_name = f"{owner}/{repo}"
    for pattern in config.allowed_repo_patterns:
        if pattern == "*/*":
            return True
        pattern_regex = pattern.replace("*", ".*")
        if re.match(f"^{pattern_regex}$", full_name):
            return True
    return False


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """Simple rate limiter using sliding window."""

    def __init__(self):
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, caller: str) -> bool:
        async with self._lock:
            now = time.time()
            window_start = now - 60

            if caller not in self._requests:
                self._requests[caller] = []

            self._requests[caller] = [t for t in self._requests[caller] if t > window_start]

            if len(self._requests[caller]) >= config.max_requests_per_minute:
                return False

            self._requests[caller].append(now)
            return True


rate_limiter = RateLimiter()


# =============================================================================
# Decorators
# =============================================================================

def protected(func):
    """
    Decorator for protected operations.

    Enforces:
    - Repository whitelist
    - Rate limiting
    - Audit logging
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        token_info = await github.get_token_info()
        caller = token_info.data.get("owner", "unknown") if token_info.success else "unknown"

        # Rate limit check
        if not await rate_limiter.check(caller):
            audit_log(func.__name__, {"reason": "rate_limited"}, caller, False)
            return ToolResult(
                success=False,
                error_code=ErrorCode.RATE_LIMITED.value,
                error_message=f"Rate limit exceeded: {config.max_requests_per_minute} requests/minute"
            ).to_response()

        # Repo whitelist check
        if "owner" in kwargs and "repo" in kwargs:
            if not check_repo_allowed(kwargs["owner"], kwargs["repo"]):
                audit_log(func.__name__, {
                    "reason": "repo_not_allowed",
                    "repo": f"{kwargs['owner']}/{kwargs['repo']}"
                }, caller, False)
                return ToolResult(
                    success=False,
                    error_code=ErrorCode.PERMISSION_DENIED.value,
                    error_message=f"Repository not in whitelist: {kwargs['owner']}/{kwargs['repo']}"
                ).to_response()

        result = await func(*args, **kwargs)

        audit_log(
            func.__name__,
            {k: v for k, v in kwargs.items() if not k.startswith("_")},
            caller,
            True
        )

        return result

    return wrapper


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
async def who_am_i() -> List[TextContent]:
    """
    Get information about the current GitHub token.

    Returns token owner, scopes, and repository counts.
    """
    result = await github.get_token_info()
    if not result.success:
        return result.to_response()

    return ToolResult(success=True, data={
        "token_owner": result.data["owner"],
        "name": result.data.get("name", ""),
        "scopes": result.data.get("scopes", []),
        "public_repos": result.data.get("public_repos", 0),
        "private_repos": result.data.get("private_repos", 0)
    }).to_response()


@mcp.tool()
async def list_my_repos(type: str = "owner") -> List[TextContent]:
    """
    List repositories owned by the token owner.

    Args:
        type: Filter type - "owner", "member", or "all"
    """
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()

    owner = token_info.data["owner"]
    result = await github.request(
        "GET",
        f"/users/{owner}/repos",
        params={"type": type, "per_page": 100, "sort": "updated"}
    )

    if not result.success:
        return result.to_response()

    repos = [{
        "full_name": r["full_name"],
        "name": r["name"],
        "private": r["private"],
        "description": r.get("description", ""),
        "updated_at": r.get("updated_at", "")
    } for r in result.data]

    return ToolResult(success=True, data={
        "owner": owner,
        "total": len(repos),
        "repositories": repos
    }).to_response()


@mcp.tool()
async def list_repos(owner: str, type: str = "owner") -> List[TextContent]:
    """
    List repositories for any user or organization.

    Args:
        owner: GitHub username or organization name
        type: Filter type - "owner", "member", or "all"
    """
    if not validate_username(owner):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid username format"
        ).to_response()

    result = await github.request(
        "GET",
        f"/users/{owner}/repos",
        params={"type": type, "per_page": 100}
    )

    if not result.success:
        return result.to_response()

    repos = [{
        "full_name": r["full_name"],
        "name": r["name"],
        "private": r["private"],
        "description": r.get("description", "")
    } for r in result.data]

    return ToolResult(success=True, data={
        "owner": owner,
        "total": len(repos),
        "repositories": repos
    }).to_response()


@mcp.tool()
async def list_collaborators(owner: str, repo: str) -> List[TextContent]:
    """
    List collaborators for a repository.

    Args:
        owner: Repository owner (user or organization)
        repo: Repository name

    Note: Requires push access to the repository.
    """
    if not validate_username(owner) or not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid owner or repository name"
        ).to_response()

    result = await github.request(
        "GET",
        f"/repos/{owner}/{repo}/collaborators",
        params={"per_page": 100}
    )

    if not result.success:
        return result.to_response()

    owner_type = await github.get_owner_type(owner)

    collaborators = [{
        "login": c["login"],
        "permissions": c.get("permissions", {}),
        "role_name": c.get("role_name", "collaborator" if owner_type == "User" else "unknown")
    } for c in result.data]

    return ToolResult(success=True, data={
        "repository": f"{owner}/{repo}",
        "owner_type": owner_type,
        "total": len(collaborators),
        "collaborators": collaborators,
        "note": "Personal repos have only owner/collaborator roles. Organizations support granular permissions."
    }).to_response()


@mcp.tool()
@protected
async def add_collaborator(
    owner: str,
    repo: str,
    username: str,
    permission: str = "push"
) -> List[TextContent]:
    """
    Add a collaborator to a repository.

    Args:
        owner: Repository owner
        repo: Repository name
        username: GitHub username to add
        permission: Permission level (for organizations only)
            - "pull": Read-only access
            - "triage": Read + manage issues/PRs
            - "push": Read + write access
            - "maintain": Push + manage repo settings
            - "admin": Full access

    Note: Personal account repositories only support collaborator access (read+write).
          The permission parameter is only effective for organization repositories.
    """
    if not validate_username(owner) or not validate_username(username):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid username format"
        ).to_response()

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    valid_permissions = {"pull", "push", "admin", "maintain", "triage"}
    if permission.lower() not in valid_permissions:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Invalid permission. Valid values: {', '.join(sorted(valid_permissions))}"
        ).to_response()

    owner_type = await github.get_owner_type(owner)

    # For personal accounts, permission parameter is ignored by GitHub API
    # Collaborators always get read+write access
    if owner_type == "User":
        result = await github.request(
            "PUT",
            f"/repos/{owner}/{repo}/collaborators/{username}"
        )
        effective_permission = "push (read+write)"
    else:
        result = await github.request(
            "PUT",
            f"/repos/{owner}/{repo}/collaborators/{username}",
            json_data={"permission": permission.lower()}
        )
        effective_permission = permission.lower()

    if not result.success:
        return result.to_response()

    github.invalidate_cache(f"{owner}/{repo}")

    response_data = {
        "action": "invitation_sent",
        "repository": f"{owner}/{repo}",
        "username": username,
        "owner_type": owner_type,
        "permission": effective_permission
    }

    if owner_type == "User":
        response_data["note"] = "Personal repos only support collaborator access (read+write). Permission parameter ignored."

    return ToolResult(success=True, data=response_data).to_response()


@mcp.tool()
@protected
async def remove_collaborator(owner: str, repo: str, username: str) -> List[TextContent]:
    """
    Remove a collaborator from a repository.

    Args:
        owner: Repository owner
        repo: Repository name
        username: GitHub username to remove
    """
    if not validate_username(owner) or not validate_username(username):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid username format"
        ).to_response()

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    result = await github.request(
        "DELETE",
        f"/repos/{owner}/{repo}/collaborators/{username}"
    )

    if not result.success:
        return result.to_response()

    github.invalidate_cache(f"{owner}/{repo}")

    return ToolResult(success=True, data={
        "action": "removed",
        "repository": f"{owner}/{repo}",
        "username": username
    }).to_response()


@mcp.tool()
async def security_status() -> List[TextContent]:
    """
    Display current security configuration.
    """
    token_info = await github.get_token_info()
    token_owner = token_info.data.get("owner", "unknown") if token_info.success else "unknown"

    return ToolResult(success=True, data={
        "token_owner": token_owner,
        "security_config": {
            "repo_whitelist": config.allowed_repo_patterns,
            "rate_limit": f"{config.max_requests_per_minute} requests/minute",
            "audit_log": config.audit_log_path
        },
        "active_protections": [
            "Input validation",
            "Repository whitelist",
            "Rate limiting",
            "Audit logging",
            "GitHub token scope enforcement"
        ]
    }).to_response()


@mcp.tool()
async def audit_log_recent(limit: int = 20) -> List[TextContent]:
    """
    Retrieve recent audit log entries.

    Args:
        limit: Maximum number of entries to return (default: 20)
    """
    try:
        with open(config.audit_log_path, "r") as f:
            lines = f.readlines()
            recent = lines[-limit:] if len(lines) > limit else lines
            logs = []
            for line in recent:
                if line.strip():
                    try:
                        logs.append(json.loads(line.split(" | ")[-1]))
                    except:
                        pass

        return ToolResult(success=True, data={
            "total": len(logs),
            "logs": logs
        }).to_response()
    except FileNotFoundError:
        return ToolResult(success=True, data={
            "total": 0,
            "logs": [],
            "note": "No audit log entries yet"
        }).to_response()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("GitHub User Management MCP Server")
    print("=" * 50)
    print(f"Repo whitelist: {config.allowed_repo_patterns}")
    print(f"Rate limit: {config.max_requests_per_minute} req/min")
    print(f"Audit log: {config.audit_log_path}")
    print("=" * 50)
    mcp.run()
