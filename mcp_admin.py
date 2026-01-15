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

# ============= Logging Configuration =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("github-user-mcp")

# ============= Server Configuration =============
@dataclass
class ServerConfig:
    name: str = "GitHub User Management Agent"
    version: str = "2.0.0"
    timeout: int = 30
    max_retries: int = 3
    rate_limit_requests: int = 30
    rate_limit_window: int = 60
    cache_ttl: int = 300
    batch_delay: float = 0.5


config = ServerConfig()

# ============= Error Types =============
class ErrorCode(Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    GITHUB_API_ERROR = "GITHUB_API_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class ToolResult:
    success: bool
    data: Optional[Dict] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_response(self) -> List[TextContent]:
        if self.success:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "data": self.data,
                    "metadata": self.metadata
                }, indent=2, ensure_ascii=False)
            )]
        else:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error_code": self.error_code,
                    "error_message": self.error_message
                }, indent=2, ensure_ascii=False)
            )]


# ============= Cache =============
class SimpleCache:
    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, tuple] = {}
        self._default_ttl = default_ttl
    
    def _make_key(self, prefix: str, *args) -> str:
        key_data = f"{prefix}:{':'.join(str(a) for a in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expiry = time.time() + (ttl or self._default_ttl)
        self._cache[key] = (value, expiry)
    
    def invalidate(self, pattern: str = "") -> int:
        count = 0
        keys_to_delete = [k for k in self._cache if pattern in k]
        for key in keys_to_delete:
            del self._cache[key]
            count += 1
        return count


# ============= Rate Limiter =============
class RateLimiter:
    def __init__(self, rate: int, window: int):
        self._rate = rate
        self._window = window
        self._tokens = rate
        self._last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self._rate, self._tokens + elapsed * (self._rate / self._window))
            self._last_update = now
            
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False
    
    async def wait_for_token(self) -> None:
        while not await self.acquire():
            await asyncio.sleep(0.1)


# ============= GitHub Client =============
class GitHubClient:
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: str):
        if not token:
            raise ValueError("GitHub token is required")
        
        self._token = token
        self._headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        self._cache = SimpleCache(default_ttl=config.cache_ttl)
        self._rate_limiter = RateLimiter(config.rate_limit_requests, config.rate_limit_window)
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=config.timeout,
                headers=self._headers,
                follow_redirects=True
            )
        return self._client
    
    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def request(
        self,
        method: str,
        endpoint: str,
        use_cache: bool = False,
        cache_ttl: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        url = f"{self.BASE_URL}{endpoint}"
        cache_key = self._cache._make_key(method, endpoint, str(kwargs.get('params', '')))
        
        if method == "GET" and use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return ToolResult(success=True, data=cached, metadata={"cached": True})
        
        await self._rate_limiter.wait_for_token()
        
        last_error = None
        for attempt in range(config.max_retries):
            try:
                client = await self._get_client()
                response = await client.request(method, url, **kwargs)
                
                if response.status_code == 403:
                    remaining = response.headers.get('X-RateLimit-Remaining', '0')
                    if remaining == '0':
                        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                        wait_time = max(0, reset_time - time.time())
                        return ToolResult(
                            success=False,
                            error_code=ErrorCode.RATE_LIMITED.value,
                            error_message=f"Rate limited. Reset in {int(wait_time)} seconds"
                        )
                
                if response.status_code == 401:
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.AUTHENTICATION_ERROR.value,
                        error_message="Invalid GitHub token"
                    )
                
                if response.status_code == 404:
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.NOT_FOUND.value,
                        error_message="Resource not found"
                    )
                
                if response.status_code >= 400:
                    error_body = response.json() if response.text else {}
                    return ToolResult(
                        success=False,
                        error_code=ErrorCode.GITHUB_API_ERROR.value,
                        error_message=error_body.get('message', f"HTTP {response.status_code}")
                    )
                
                data = response.json() if response.text else {}
                
                if method == "GET" and use_cache:
                    self._cache.set(cache_key, data, cache_ttl)
                
                return ToolResult(
                    success=True,
                    data=data,
                    metadata={
                        "rate_limit_remaining": response.headers.get('X-RateLimit-Remaining'),
                        "cached": False
                    }
                )
                
            except httpx.TimeoutException:
                last_error = "Request timeout"
                logger.warning(f"Timeout on attempt {attempt + 1}/{config.max_retries}")
            except httpx.NetworkError as e:
                last_error = f"Network error: {str(e)}"
                logger.warning(f"Network error on attempt {attempt + 1}/{config.max_retries}")
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error: {e}")
                break
            
            if attempt < config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        return ToolResult(
            success=False,
            error_code=ErrorCode.NETWORK_ERROR.value,
            error_message=last_error or "Request failed after retries"
        )
    
    def invalidate_cache(self, pattern: str = "") -> int:
        return self._cache.invalidate(pattern)


# ============= Input Validation =============
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


# ============= Initialize MCP Server =============
mcp = FastMCP(config.name)

GITHUB_TOKEN = os.environ.get("MY_GITHUB_TOKEN")
if not GITHUB_TOKEN:
    logger.warning("MY_GITHUB_TOKEN not set")
    github_client = None
else:
    github_client = GitHubClient(GITHUB_TOKEN)


def require_client(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if github_client is None:
            return ToolResult(
                success=False,
                error_code=ErrorCode.AUTHENTICATION_ERROR.value,
                error_message="GitHub token not configured. Set MY_GITHUB_TOKEN environment variable."
            ).to_response()
        return await func(*args, **kwargs)
    return wrapper


# ============= MCP Tools =============

@mcp.tool()
@require_client
async def list_repos(
    visibility: str = "all",
    sort: str = "updated",
    per_page: int = 30
) -> List[TextContent]:
    """
    Liet ke tat ca repositories ma nguoi dung co quyen truy cap.
    List all GitHub repositories accessible to the authenticated user.
    
    Dung khi can / Use when you need to:
    - Xem danh sach repos / See all accessible repos
    - Tim repo trong danh sach / Find a specific repo
    - Xem tong quan tai khoan GitHub / Get account overview
    
    Args:
        visibility: "all" (tat ca), "public" (cong khai), "private" (rieng tu)
        sort: "created", "updated", "pushed", "full_name"
        per_page: So luong ket qua (max 100) / Number of results
    
    Returns:
        Danh sach repos voi ten, owner, trang thai / List of repos with name, owner, status
    
    Examples:
        - Liet ke tat ca: list_repos()
        - Chi private repos: list_repos(visibility="private")
        - Sap xep theo ten: list_repos(sort="full_name")
    """
    if visibility not in ["all", "public", "private"]:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="visibility must be 'all', 'public', or 'private'"
        ).to_response()
    
    if sort not in ["created", "updated", "pushed", "full_name"]:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="sort must be 'created', 'updated', 'pushed', or 'full_name'"
        ).to_response()
    
    per_page = min(max(1, per_page), 100)
    
    result = await github_client.request(
        "GET", "/user/repos",
        use_cache=True, cache_ttl=60,
        params={
            "visibility": visibility,
            "affiliation": "owner,collaborator,organization_member",
            "sort": sort,
            "per_page": per_page
        }
    )
    
    if not result.success:
        return result.to_response()
    
    repos = [{
        "full_name": r["full_name"],
        "owner": r["owner"]["login"],
        "name": r["name"],
        "private": r["private"],
        "permissions": r.get("permissions", {}),
        "default_branch": r.get("default_branch"),
        "updated_at": r.get("updated_at")
    } for r in result.data]
    
    return ToolResult(
        success=True,
        data={"total_count": len(repos), "repositories": repos},
        metadata=result.metadata
    ).to_response()


@mcp.tool()
@require_client
async def list_collaborators(
    owner: str,
    repo: str,
    permission_filter: str = "all"
) -> List[TextContent]:
    """
    Liet ke collaborators cua repository voi quyen chi tiet.
    List all collaborators of a repository with detailed permissions.
    
    Dung khi can / Use when you need to:
    - Xem ai co quyen truy cap repo / See who has access
    - Kiem tra quyen nguoi dung / Check user permissions
    - Audit quyen truy cap / Audit repository access
    
    Args:
        owner: Chu so huu repo / Repository owner (e.g., "octocat")
        repo: Ten repo / Repository name (e.g., "hello-world")
        permission_filter: Loc theo quyen / Filter by permission
            - "all": tat ca / all collaborators
            - "pull": chi doc / read only
            - "push": ghi / write
            - "admin": quan tri / admin
            - "maintain", "triage"
    
    Returns:
        Danh sach collaborators voi username va quyen / List with usernames and permissions
    
    Examples:
        - Tat ca collaborators: list_collaborators("myorg", "myrepo")
        - Chi admins: list_collaborators("myorg", "myrepo", permission_filter="admin")
        - Chi nguoi co quyen ghi: list_collaborators("myorg", "myrepo", permission_filter="push")
    """
    if not validate_username(owner):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Invalid owner username: {owner}"
        ).to_response()
    
    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()
    
    params = {"per_page": 100}
    if permission_filter != "all":
        params["permission"] = permission_filter
    
    result = await github_client.request(
        "GET", f"/repos/{owner}/{repo}/collaborators",
        use_cache=True, cache_ttl=120,
        params=params
    )
    
    if not result.success:
        return result.to_response()
    
    collaborators = [{
        "username": u["login"],
        "id": u["id"],
        "permissions": u.get("permissions", {}),
        "role_name": u.get("role_name", "unknown")
    } for u in result.data]
    
    return ToolResult(
        success=True,
        data={
            "repository": f"{owner}/{repo}",
            "total_count": len(collaborators),
            "collaborators": collaborators
        },
        metadata=result.metadata
    ).to_response()


@mcp.tool()
@require_client
async def check_permission(
    owner: str,
    repo: str,
    username: str
) -> List[TextContent]:
    """
    Kiem tra quyen cua nguoi dung trong repository.
    Check specific permission level of a user in a repository.
    
    Dung khi can / Use when you need to:
    - Xac nhan nguoi dung co quyen khong / Verify user access
    - Kiem tra cap quyen hien tai / Check current permission level
    - Xac nhan truoc khi thay doi quyen / Confirm before modifying
    
    Args:
        owner: Chu so huu repo / Repository owner
        repo: Ten repo / Repository name
        username: Ten nguoi dung can kiem tra / Username to check
    
    Returns:
        Thong tin quyen va trang thai truy cap / Permission info and access status
    
    Examples:
        - Kiem tra quyen: check_permission("myorg", "myrepo", "johndoe")
        - Xem user co quyen admin khong: check_permission("company", "project", "alice")
    """
    for label, value in [("owner", owner), ("username", username)]:
        if not validate_username(value):
            return ToolResult(
                success=False,
                error_code=ErrorCode.VALIDATION_ERROR.value,
                error_message=f"Invalid {label}: {value}"
            ).to_response()
    
    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()
    
    result = await github_client.request(
        "GET", f"/repos/{owner}/{repo}/collaborators/{username}/permission",
        use_cache=True, cache_ttl=60
    )
    
    if not result.success:
        if result.error_code == ErrorCode.NOT_FOUND.value:
            return ToolResult(
                success=True,
                data={
                    "username": username,
                    "repository": f"{owner}/{repo}",
                    "has_access": False,
                    "permission": None
                }
            ).to_response()
        return result.to_response()
    
    return ToolResult(
        success=True,
        data={
            "username": username,
            "repository": f"{owner}/{repo}",
            "has_access": True,
            "permission": result.data.get("permission"),
            "role_name": result.data.get("role_name")
        },
        metadata=result.metadata
    ).to_response()


@mcp.tool()
@require_client
async def add_collaborator(
    owner: str,
    repo: str,
    username: str,
    permission: str = "pull"
) -> List[TextContent]:
    """
    Them nguoi dung vao repository voi quyen chi dinh.
    Add a user as collaborator with specified permission level.
    
    Luu y: Se gui email moi cho nguoi dung.
    Note: This sends an invitation email to the user.
    
    Dung khi can / Use when you need to:
    - Them nguoi moi vao repo / Add new user to repo
    - Moi nguoi cong tac / Invite collaborator
    - Cap quyen truy cap / Grant repository access
    
    Args:
        owner: Chu so huu repo / Repository owner
        repo: Ten repo / Repository name
        username: Ten nguoi dung can them / Username to add
        permission: Cap quyen / Permission level
            - "pull": chi doc / read only
            - "push": doc va ghi / read and write
            - "admin": toan quyen / full access
            - "maintain": bao tri / maintainer
            - "triage": phan loai issues / triage issues
    
    Returns:
        Xac nhan da gui loi moi / Confirmation of invitation sent
    
    Examples:
        - Them voi quyen doc: add_collaborator("myorg", "myrepo", "newuser", "pull")
        - Them voi quyen ghi: add_collaborator("myorg", "myrepo", "dev1", "push")
        - Them admin: add_collaborator("myorg", "myrepo", "lead", "admin")
    """
    for label, value in [("owner", owner), ("username", username)]:
        if not validate_username(value):
            return ToolResult(
                success=False,
                error_code=ErrorCode.VALIDATION_ERROR.value,
                error_message=f"Invalid {label}: {value}"
            ).to_response()
    
    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()
    
    if not validate_permission(permission):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Invalid permission: {permission}. Must be: pull, push, admin, maintain, or triage"
        ).to_response()
    
    result = await github_client.request(
        "PUT", f"/repos/{owner}/{repo}/collaborators/{username}",
        json={"permission": permission.lower()}
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
            "message": f"Invitation sent to {username} for {owner}/{repo} with '{permission}' permission"
        }
    ).to_response()


@mcp.tool()
@require_client
async def remove_collaborator(
    owner: str,
    repo: str,
    username: str
) -> List[TextContent]:
    """
    Xoa nguoi dung khoi repository, thu hoi quyen truy cap.
    Remove a collaborator from repository, revoking their access.
    
    Luu y: Hanh dong nay co hieu luc ngay lap tuc.
    Note: This action is immediate and cannot be undone automatically.
    
    Dung khi can / Use when you need to:
    - Xoa nguoi dung khoi repo / Remove user from repo
    - Thu hoi quyen truy cap / Revoke access
    - Don dep quyen / Clean up permissions
    
    Args:
        owner: Chu so huu repo / Repository owner
        repo: Ten repo / Repository name
        username: Ten nguoi dung can xoa / Username to remove
    
    Returns:
        Xac nhan da xoa / Confirmation of removal
    
    Examples:
        - Xoa user: remove_collaborator("myorg", "myrepo", "formeruser")
        - Thu hoi quyen: remove_collaborator("company", "project", "exemployee")
    """
    for label, value in [("owner", owner), ("username", username)]:
        if not validate_username(value):
            return ToolResult(
                success=False,
                error_code=ErrorCode.VALIDATION_ERROR.value,
                error_message=f"Invalid {label}: {value}"
            ).to_response()
    
    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()
    
    result = await github_client.request(
        "DELETE", f"/repos/{owner}/{repo}/collaborators/{username}"
    )
    
    if not result.success:
        return result.to_response()
    
    github_client.invalidate_cache(f"{owner}/{repo}")
    
    return ToolResult(
        success=True,
        data={
            "action": "removed",
            "username": username,
            "repository": f"{owner}/{repo}",
            "message": f"Successfully removed {username} from {owner}/{repo}"
        }
    ).to_response()


@mcp.tool()
@require_client
async def update_permission(
    owner: str,
    repo: str,
    username: str,
    new_permission: str
) -> List[TextContent]:
    """
    Cap nhat quyen cua collaborator trong repository.
    Update permission level of an existing collaborator.
    
    Dung khi can / Use when you need to:
    - Nang cap quyen / Promote user (e.g., pull -> push)
    - Ha cap quyen / Demote user (e.g., admin -> push)
    - Thay doi quyen / Adjust permission level
    
    Args:
        owner: Chu so huu repo / Repository owner
        repo: Ten repo / Repository name
        username: Ten collaborator / Username of existing collaborator
        new_permission: Quyen moi / New permission level
            - "pull", "push", "admin", "maintain", "triage"
    
    Returns:
        Xac nhan da cap nhat / Confirmation of update
    
    Examples:
        - Nang cap len admin: update_permission("myorg", "myrepo", "user1", "admin")
        - Ha cap xuong read-only: update_permission("myorg", "myrepo", "user1", "pull")
    """
    return await add_collaborator(owner, repo, username, new_permission)


@mcp.tool()
@require_client
async def bulk_add_collaborators(
    owner: str,
    repo: str,
    usernames: List[str],
    permission: str = "pull"
) -> List[TextContent]:
    """
    Them nhieu nguoi dung vao repository cung luc.
    Add multiple users as collaborators in a single operation.
    
    Luu y: Se gui email moi cho tat ca nguoi dung.
    Note: This sends invitation emails to all users.
    
    Dung khi can / Use when you need to:
    - Them ca nhom / Add a team to repo
    - Them nhieu nguoi cung luc / Add multiple users at once
    - Thiet lap du an moi / Set up new project with team
    
    Args:
        owner: Chu so huu repo / Repository owner
        repo: Ten repo / Repository name
        usernames: Danh sach usernames (toi da 50) / List of usernames (max 50)
        permission: Quyen cho tat ca / Permission for all users
    
    Returns:
        Tong hop thanh cong va that bai / Summary of successes and failures
    
    Examples:
        - Them nhom dev: bulk_add_collaborators("myorg", "myrepo", ["dev1", "dev2", "dev3"], "push")
        - Them nhom doc: bulk_add_collaborators("myorg", "docs", ["reader1", "reader2"], "pull")
    """
    if not usernames:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="usernames list cannot be empty"
        ).to_response()
    
    if len(usernames) > 50:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Maximum 50 users per bulk operation"
        ).to_response()
    
    if not validate_permission(permission):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Invalid permission: {permission}"
        ).to_response()
    
    invalid = [u for u in usernames if not validate_username(u)]
    if invalid:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Invalid usernames: {', '.join(invalid)}"
        ).to_response()
    
    results = {"successful": [], "failed": []}
    
    for i, username in enumerate(usernames):
        logger.info(f"Adding collaborator {i+1}/{len(usernames)}: {username}")
        
        result = await github_client.request(
            "PUT", f"/repos/{owner}/{repo}/collaborators/{username}",
            json={"permission": permission.lower()}
        )
        
        if result.success:
            results["successful"].append(username)
        else:
            results["failed"].append({"username": username, "error": result.error_message})
        
        if i < len(usernames) - 1:
            await asyncio.sleep(config.batch_delay)
    
    github_client.invalidate_cache(f"{owner}/{repo}")
    
    return ToolResult(
        success=True,
        data={
            "repository": f"{owner}/{repo}",
            "permission": permission,
            "total": len(usernames),
            "success_count": len(results["successful"]),
            "failed_count": len(results["failed"]),
            "successful": results["successful"],
            "failed": results["failed"]
        }
    ).to_response()


@mcp.tool()
@require_client
async def bulk_remove_collaborators(
    owner: str,
    repo: str,
    usernames: List[str]
) -> List[TextContent]:
    """
    Xoa nhieu nguoi dung khoi repository cung luc.
    Remove multiple collaborators in a single operation.
    
    Luu y: Hanh dong nay co hieu luc ngay lap tuc.
    Note: This action is immediate.
    
    Dung khi can / Use when you need to:
    - Xoa ca nhom / Remove a team from repo
    - Thu hoi quyen hang loat / Revoke access from multiple users
    - Don dep quyen / Clean up permissions in bulk
    
    Args:
        owner: Chu so huu repo / Repository owner
        repo: Ten repo / Repository name
        usernames: Danh sach can xoa (toi da 50) / List of usernames to remove (max 50)
    
    Returns:
        Tong hop thanh cong va that bai / Summary of successes and failures
    
    Examples:
        - Xoa nhom: bulk_remove_collaborators("myorg", "myrepo", ["user1", "user2"])
    """
    if not usernames:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="usernames list cannot be empty"
        ).to_response()
    
    if len(usernames) > 50:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Maximum 50 users per bulk operation"
        ).to_response()
    
    results = {"successful": [], "failed": []}
    
    for i, username in enumerate(usernames):
        logger.info(f"Removing collaborator {i+1}/{len(usernames)}: {username}")
        
        result = await github_client.request(
            "DELETE", f"/repos/{owner}/{repo}/collaborators/{username}"
        )
        
        if result.success:
            results["successful"].append(username)
        else:
            results["failed"].append({"username": username, "error": result.error_message})
        
        if i < len(usernames) - 1:
            await asyncio.sleep(config.batch_delay)
    
    github_client.invalidate_cache(f"{owner}/{repo}")
    
    return ToolResult(
        success=True,
        data={
            "repository": f"{owner}/{repo}",
            "total": len(usernames),
            "success_count": len(results["successful"]),
            "failed_count": len(results["failed"]),
            "successful": results["successful"],
            "failed": results["failed"]
        }
    ).to_response()


@mcp.tool()
@require_client
async def find_user_repos(
    username: str,
    repos_to_search: Optional[List[str]] = None
) -> List[TextContent]:
    """
    Tim nguoi dung co quyen trong nhung repositories nao.
    Find which repositories a user has access to.
    
    Dung khi can / Use when you need to:
    - Kiem tra quyen cua user qua nhieu repos / Audit user access across repos
    - Tim tat ca repos user co quyen / Find all repos user can access
    - Chuan bi offboarding / Prepare for offboarding
    
    Args:
        username: Ten nguoi dung can tim / Username to search for
        repos_to_search: Danh sach repos de tim (optional) / List of repos to search
            - Neu khong chi dinh, tim trong tat ca repos / If not provided, searches all accessible repos
            - Dinh dang "owner/repo" / Format "owner/repo"
    
    Returns:
        Danh sach repos va quyen tuong ung / List of repos with permission levels
    
    Examples:
        - Tim trong tat ca repos: find_user_repos("johndoe")
        - Tim trong repos cu the: find_user_repos("johndoe", ["myorg/repo1", "myorg/repo2"])
    """
    if not validate_username(username):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Invalid username: {username}"
        ).to_response()
    
    if repos_to_search:
        repos = []
        for repo_full_name in repos_to_search:
            if "/" not in repo_full_name:
                continue
            owner, name = repo_full_name.split("/", 1)
            repos.append({"owner": owner, "name": name, "full_name": repo_full_name})
    else:
        repos_result = await github_client.request(
            "GET", "/user/repos",
            use_cache=True,
            params={"per_page": 100}
        )
        if not repos_result.success:
            return repos_result.to_response()
        repos = [{"owner": r["owner"]["login"], "name": r["name"], "full_name": r["full_name"]} 
                 for r in repos_result.data]
    
    found_in = []
    errors = []
    
    for i, repo in enumerate(repos):
        logger.info(f"Checking repo {i+1}/{len(repos)}: {repo['full_name']}")
        
        result = await github_client.request(
            "GET", f"/repos/{repo['owner']}/{repo['name']}/collaborators/{username}/permission",
            use_cache=True, cache_ttl=120
        )
        
        if result.success:
            found_in.append({
                "repository": repo['full_name'],
                "permission": result.data.get("permission"),
                "role_name": result.data.get("role_name")
            })
        elif result.error_code != ErrorCode.NOT_FOUND.value:
            errors.append({"repository": repo['full_name'], "error": result.error_message})
        
        if i < len(repos) - 1:
            await asyncio.sleep(0.2)
    
    return ToolResult(
        success=True,
        data={
            "username": username,
            "repos_searched": len(repos),
            "repos_with_access": len(found_in),
            "repositories": found_in,
            "errors": errors if errors else None
        }
    ).to_response()


@mcp.tool()
@require_client
async def audit_repo(owner: str, repo: str) -> List[TextContent]:
    """
    Tao bao cao audit quyen truy cap repository.
    Generate comprehensive permission audit report for a repository.
    
    Dung khi can / Use when you need to:
    - Xem tong quan quyen truy cap / Review all access
    - Chuan bi security audit / Prepare for security audit
    - Tao tai lieu quyen / Document permissions
    - Tim van de quyen / Identify permission issues
    
    Args:
        owner: Chu so huu repo / Repository owner
        repo: Ten repo / Repository name
    
    Returns:
        Bao cao chi tiet theo nhom quyen / Detailed report grouped by permission level
    
    Examples:
        - Audit repo: audit_repo("myorg", "myrepo")
        - Kiem tra bao mat: audit_repo("company", "sensitive-project")
    """
    if not validate_username(owner):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Invalid owner: {owner}"
        ).to_response()
    
    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()
    
    repo_result = await github_client.request("GET", f"/repos/{owner}/{repo}", use_cache=True)
    if not repo_result.success:
        return repo_result.to_response()
    
    collab_result = await github_client.request(
        "GET", f"/repos/{owner}/{repo}/collaborators",
        use_cache=True, params={"per_page": 100}
    )
    if not collab_result.success:
        return collab_result.to_response()
    
    by_permission = {"admin": [], "maintain": [], "push": [], "triage": [], "pull": []}
    
    for collab in collab_result.data:
        role = collab.get("role_name", "unknown")
        if role in by_permission:
            by_permission[role].append({"username": collab["login"], "id": collab["id"]})
    
    return ToolResult(
        success=True,
        data={
            "repository": f"{owner}/{repo}",
            "visibility": "private" if repo_result.data.get("private") else "public",
            "default_branch": repo_result.data.get("default_branch"),
            "audit_timestamp": datetime.utcnow().isoformat(),
            "total_collaborators": len(collab_result.data),
            "by_permission": by_permission,
            "summary": {
                "admins": len(by_permission["admin"]),
                "maintainers": len(by_permission["maintain"]),
                "writers": len(by_permission["push"]),
                "triagers": len(by_permission["triage"]),
                "readers": len(by_permission["pull"])
            }
        }
    ).to_response()


@mcp.tool()
@require_client
async def sync_collaborators(
    source_repo: str,
    target_repos: List[str],
    permission: str = "pull",
    dry_run: bool = False
) -> List[TextContent]:
    """
    Dong bo collaborators tu repo nguon sang cac repos dich.
    Synchronize collaborators from source repository to target repositories.
    
    Dung khi can / Use when you need to:
    - Sao chep quyen tu repo nay sang repo khac / Mirror access between repos
    - Thiet lap quyen nhat quan / Set up consistent permissions
    - Clone quyen cho du an moi / Clone team access to new repo
    
    Args:
        source_repo: Repo nguon "owner/repo" / Source repository
        target_repos: Danh sach repos dich / List of target repositories
        permission: Quyen ap dung cho repos dich / Permission for target repos
        dry_run: Neu True, chi xem truoc / If True, only preview changes
    
    Returns:
        Ket qua dong bo hoac xem truoc / Sync results or preview
    
    Examples:
        - Dong bo: sync_collaborators("myorg/main", ["myorg/sub1", "myorg/sub2"], "push")
        - Xem truoc: sync_collaborators("myorg/main", ["myorg/sub1"], dry_run=True)
    """
    if "/" not in source_repo:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="source_repo must be in 'owner/repo' format"
        ).to_response()
    
    if not target_repos:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="target_repos cannot be empty"
        ).to_response()
    
    if len(target_repos) > 10:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Maximum 10 target repos per sync operation"
        ).to_response()
    
    source_owner, source_name = source_repo.split("/", 1)
    
    source_result = await github_client.request(
        "GET", f"/repos/{source_owner}/{source_name}/collaborators",
        use_cache=True, params={"per_page": 100}
    )
    if not source_result.success:
        return source_result.to_response()
    
    source_users = [c["login"] for c in source_result.data]
    
    if dry_run:
        return ToolResult(
            success=True,
            data={
                "dry_run": True,
                "source_repo": source_repo,
                "target_repos": target_repos,
                "users_to_sync": source_users,
                "permission": permission,
                "message": f"Would add {len(source_users)} users to {len(target_repos)} repos"
            }
        ).to_response()
    
    sync_results = {}
    
    for target_repo in target_repos:
        if "/" not in target_repo:
            sync_results[target_repo] = {"error": "Invalid repo format"}
            continue
        
        target_owner, target_name = target_repo.split("/", 1)
        results = {"successful": [], "failed": []}
        
        for username in source_users:
            result = await github_client.request(
                "PUT", f"/repos/{target_owner}/{target_name}/collaborators/{username}",
                json={"permission": permission}
            )
            
            if result.success:
                results["successful"].append(username)
            else:
                results["failed"].append({"username": username, "error": result.error_message})
            
            await asyncio.sleep(config.batch_delay)
        
        sync_results[target_repo] = results
        github_client.invalidate_cache(target_repo)
    
    return ToolResult(
        success=True,
        data={
            "source_repo": source_repo,
            "source_users": source_users,
            "permission": permission,
            "sync_results": sync_results
        }
    ).to_response()


# ============= Main =============
if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        async def run_tests():
            print("Testing GitHub User Management MCP Server\n")
            print("=" * 50)
            
            if not GITHUB_TOKEN:
                print("ERROR: MY_GITHUB_TOKEN not set")
                return
            
            print("\nTest 1: List repos")
            response = await list_repos(per_page=5)
            print(response[0].text[:500])
            
            print("\nTest 2: Validation")
            response = await add_collaborator("invalid@user", "repo", "user")
            print(response[0].text)
            
            print("\n" + "=" * 50)
            print("Tests completed!")
        
        asyncio.run(run_tests())
    else:
        mcp.run()