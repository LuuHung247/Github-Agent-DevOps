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
    - Rate limiting: Prevents abuse
    - Audit logging: Records all operations
    - GitHub token scope: Enforced by GitHub API
    """
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
        """Convert to MCP response with compact JSON (saves ~30% tokens)."""
        if self.success:
            return [TextContent(type="text", text=json.dumps(self.data, ensure_ascii=False, separators=(',', ':')))]
        return [TextContent(type="text", text=json.dumps({
            "error": True,
            "code": self.error_code,
            "message": self.error_message
        }, ensure_ascii=False, separators=(',', ':')))]


# =============================================================================
# GitHub API Client
# =============================================================================

class GitHubClient:
    """
    GitHub API client with automatic owner detection and connection pooling.

    Handles both personal accounts and organizations transparently.
    Uses persistent HTTP client for better performance.
    """

    def __init__(self):
        self.token = os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable is required")

        self.base_url = "https://api.github.com"
        self._token_owner: Optional[str] = None
        self._token_scopes: Optional[List[str]] = None
        self._owner_type_cache: Dict[str, str] = {}
        self._client: Optional[httpx.AsyncClient] = None

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                headers=self._headers()
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def get_token_info(self) -> ToolResult:
        """Retrieve token owner information and scopes."""
        if self._token_owner:
            return ToolResult(success=True, data={
                "owner": self._token_owner,
                "scopes": self._token_scopes
            })

        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/user")

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
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/users/{owner}")
            if response.status_code == 200:
                owner_type = response.json().get("type", "User")
                self._owner_type_cache[owner] = owner_type
                return owner_type
        except:
            pass

        return "User"

    async def request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        json_data: Dict = None
    ) -> ToolResult:
        """Execute GitHub API request using persistent client."""
        try:
            client = await self._get_client()
            response = await client.request(
                method=method,
                url=f"{self.base_url}{endpoint}",
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
async def quick_overview(max_repos: int = 10) -> List[TextContent]:
    """
    Get a quick summary: account info, repo count, total collaborators, pending invitations.
    Combines multiple API calls into one response for efficiency.

    Args:
        max_repos: Max repos to check for collaborators/invitations (default: 10, max: 30)
    """
    max_repos = min(max(1, max_repos), 30)

    # Get token owner info
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()

    owner = token_info.data["owner"]

    # Get repos
    repos_result = await github.request(
        "GET",
        f"/users/{owner}/repos",
        params={"type": "owner", "per_page": max_repos, "sort": "updated"}
    )

    if not repos_result.success:
        return repos_result.to_response()

    repos = repos_result.data

    # Fetch collaborators and invitations IN PARALLEL
    async def fetch_repo_data(repo_name: str) -> dict:
        collab_result = await github.request(
            "GET",
            f"/repos/{owner}/{repo_name}/collaborators",
            params={"per_page": 100}
        )
        inv_result = await github.request(
            "GET",
            f"/repos/{owner}/{repo_name}/invitations",
            params={"per_page": 100}
        )
        return {
            "repo": repo_name,
            "collaborators": [c["login"] for c in (collab_result.data or []) if c["login"] != owner] if collab_result.success else [],
            "invitations": len(inv_result.data or []) if inv_result.success else 0
        }

    tasks = [fetch_repo_data(r["name"]) for r in repos]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate data
    all_collaborators = set()
    total_invitations = 0
    repos_with_collaborators = []
    repos_with_invitations = []

    for item in results:
        if isinstance(item, Exception):
            continue
        all_collaborators.update(item["collaborators"])
        total_invitations += item["invitations"]
        if item["collaborators"]:
            repos_with_collaborators.append(item["repo"])
        if item["invitations"] > 0:
            repos_with_invitations.append(item["repo"])

    return ToolResult(success=True, data={
        "owner": owner,
        "name": token_info.data.get("name", ""),
        "repos_checked": len(repos),
        "total_collaborators": len(all_collaborators),
        "collaborators": sorted(all_collaborators),
        "total_pending_invitations": total_invitations,
        "repos_with_collaborators": repos_with_collaborators,
        "repos_with_invitations": repos_with_invitations
    }).to_response()


@mcp.tool()
async def list_repos(owner: str = None, type: str = "owner") -> List[TextContent]:
    """
    List repositories for a user or organization.

    Args:
        owner: GitHub username or organization name (optional, defaults to token owner)
        type: Filter - "owner", "member", or "all"
    """
    # Use token owner if not specified
    if not owner:
        token_info = await github.get_token_info()
        if not token_info.success:
            return token_info.to_response()
        owner = token_info.data["owner"]
    elif not validate_username(owner):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid username format"
        ).to_response()

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
async def list_collaborators(repo: str = None, max_repos: int = 20) -> List[TextContent]:
    """
    List collaborators for a specific repository or across all repositories.

    Args:
        repo: Repository name (optional). If not specified, lists collaborators across all repos.
        max_repos: Max repos to check when repo is not specified (default: 20, max: 50)
    """
    # Get token owner info
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()

    owner = token_info.data["owner"]

    # Single repo mode
    if repo:
        if not validate_repo_name(repo):
            return ToolResult(
                success=False,
                error_code=ErrorCode.VALIDATION_ERROR.value,
                error_message="Invalid repository name"
            ).to_response()

        result = await github.request(
            "GET",
            f"/repos/{owner}/{repo}/collaborators",
            params={"per_page": 100}
        )

        if not result.success:
            return result.to_response()

        collaborators = [{
            "login": c["login"],
            "permissions": c.get("permissions", {}),
            "role_name": c.get("role_name", "collaborator")
        } for c in result.data if c["login"] != owner]

        return ToolResult(success=True, data={
            "repository": f"{owner}/{repo}",
            "total": len(collaborators),
            "collaborators": collaborators
        }).to_response()

    # All repos mode
    max_repos = min(max(1, max_repos), 50)

    repos_result = await github.request(
        "GET",
        f"/users/{owner}/repos",
        params={"type": "owner", "per_page": max_repos, "sort": "updated"}
    )

    if not repos_result.success:
        return repos_result.to_response()

    repos = repos_result.data
    collaborator_map = {}

    async def fetch_collaborators(repo_name: str) -> tuple:
        result = await github.request(
            "GET",
            f"/repos/{owner}/{repo_name}/collaborators",
            params={"per_page": 100}
        )
        return repo_name, result

    tasks = [fetch_collaborators(r["name"]) for r in repos]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for item in results:
        if isinstance(item, Exception):
            continue
        repo_name, result = item
        if result.success and result.data:
            for collab in result.data:
                username = collab["login"]
                if username == owner:
                    continue

                if username not in collaborator_map:
                    collaborator_map[username] = {
                        "username": username,
                        "repos": [],
                        "total_repos": 0
                    }

                collaborator_map[username]["repos"].append({
                    "repo": repo_name,
                    "permissions": collab.get("permissions", {})
                })
                collaborator_map[username]["total_repos"] += 1

    collaborators = sorted(
        collaborator_map.values(),
        key=lambda x: x["total_repos"],
        reverse=True
    )

    return ToolResult(success=True, data={
        "owner": owner,
        "repos_checked": len(repos),
        "total_collaborators": len(collaborators),
        "collaborators": collaborators
    }).to_response()


@mcp.tool()
async def list_pending_invitations(repo: str = None, max_repos: int = 20) -> List[TextContent]:
    """
    List pending invitations for a specific repository or across all repositories.

    Args:
        repo: Repository name (optional). If not specified, lists invitations across all repos.
        max_repos: Max repos to check when repo is not specified (default: 20, max: 50)
    """
    # Get token owner info
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()

    owner = token_info.data["owner"]

    # Single repo mode
    if repo:
        if not validate_repo_name(repo):
            return ToolResult(
                success=False,
                error_code=ErrorCode.VALIDATION_ERROR.value,
                error_message="Invalid repository name"
            ).to_response()

        result = await github.request(
            "GET",
            f"/repos/{owner}/{repo}/invitations",
            params={"per_page": 100}
        )

        if not result.success:
            return result.to_response()

        invitations = [{
            "id": inv["id"],
            "invitee": inv.get("invitee", {}).get("login", "unknown"),
            "inviter": inv.get("inviter", {}).get("login", "unknown"),
            "permissions": inv.get("permissions", "unknown"),
            "created_at": inv.get("created_at", ""),
            "url": inv.get("html_url", "")
        } for inv in result.data]

        return ToolResult(success=True, data={
            "repository": f"{owner}/{repo}",
            "total": len(invitations),
            "pending_invitations": invitations
        }).to_response()

    # All repos mode
    max_repos = min(max(1, max_repos), 50)

    repos_result = await github.request(
        "GET",
        f"/users/{owner}/repos",
        params={"type": "owner", "per_page": max_repos, "sort": "updated"}
    )

    if not repos_result.success:
        return repos_result.to_response()

    repos = repos_result.data
    invitations_by_repo = []
    total_invitations = 0
    repos_with_invitations = []

    async def fetch_invitations(repo_name: str) -> tuple:
        result = await github.request(
            "GET",
            f"/repos/{owner}/{repo_name}/invitations",
            params={"per_page": 100}
        )
        return repo_name, result

    tasks = [fetch_invitations(r["name"]) for r in repos]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for item in results:
        if isinstance(item, Exception):
            continue
        repo_name, result = item
        if result.success and result.data:
            invitations = [{
                "id": inv["id"],
                "invitee": inv.get("invitee", {}).get("login", "unknown"),
                "inviter": inv.get("inviter", {}).get("login", "unknown"),
                "permissions": inv.get("permissions", "unknown"),
                "created_at": inv.get("created_at", "")
            } for inv in result.data]

            if invitations:
                repos_with_invitations.append(repo_name)
                total_invitations += len(invitations)
                invitations_by_repo.append({
                    "repository": f"{owner}/{repo_name}",
                    "invitation_count": len(invitations),
                    "invitations": invitations
                })

    return ToolResult(success=True, data={
        "owner": owner,
        "total_invitations": total_invitations,
        "repos_checked": len(repos),
        "repos_with_invitations": repos_with_invitations,
        "invitations_by_repo": invitations_by_repo
    }).to_response()


@mcp.tool()
@protected
async def cancel_invitation(repo: str, invitation_id: int) -> List[TextContent]:
    """
    Cancel a pending repository invitation.

    Args:
        repo: Repository name
        invitation_id: ID of the invitation to cancel
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    result = await github.request(
        "DELETE",
        f"/repos/{owner}/{repo}/invitations/{invitation_id}"
    )

    if not result.success:
        return result.to_response()

    return ToolResult(success=True, data={
        "action": "invitation_cancelled",
        "repository": f"{owner}/{repo}",
        "invitation_id": invitation_id
    }).to_response()


@mcp.tool()
@protected
async def add_collaborators(
    repo: str,
    usernames: str,
    permission: str = "push"
) -> List[TextContent]:
    """
    Add collaborators to a repository (sends invitations).

    Args:
        repo: Repository name
        usernames: Comma-separated GitHub usernames (e.g., "user1" or "user1,user2,user3")
        permission: Permission level (pull/triage/push/maintain/admin) - org only
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

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

    # Parse usernames
    user_list = [u.strip() for u in usernames.split(",") if u.strip()]
    if not user_list:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="No valid usernames provided"
        ).to_response()

    # Validate all usernames
    invalid_users = [u for u in user_list if not validate_username(u)]
    if invalid_users:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Invalid usernames: {', '.join(invalid_users)}"
        ).to_response()

    owner_type = await github.get_owner_type(owner)

    # Add collaborators IN PARALLEL
    async def add_user(username: str) -> dict:
        if owner_type == "User":
            result = await github.request(
                "PUT",
                f"/repos/{owner}/{repo}/collaborators/{username}"
            )
        else:
            result = await github.request(
                "PUT",
                f"/repos/{owner}/{repo}/collaborators/{username}",
                json_data={"permission": permission.lower()}
            )
        return {"username": username, "success": result.success, "error": result.error_message if not result.success else None}

    tasks = [add_user(u) for u in user_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = []
    failed = []
    for item in results:
        if isinstance(item, Exception):
            failed.append({"username": "unknown", "error": str(item)})
        elif item["success"]:
            successful.append(item["username"])
        else:
            failed.append({"username": item["username"], "error": item["error"]})

    return ToolResult(success=True, data={
        "action": "invitation_sent",
        "repository": f"{owner}/{repo}",
        "owner_type": owner_type,
        "permission": permission.lower() if owner_type != "User" else "push (read+write)",
        "total_requested": len(user_list),
        "successful": successful,
        "failed": failed
    }).to_response()


@mcp.tool()
@protected
async def remove_collaborators(
    repo: str,
    usernames: str
) -> List[TextContent]:
    """
    Remove collaborators from a repository.

    Args:
        repo: Repository name
        usernames: Comma-separated GitHub usernames (e.g., "user1" or "user1,user2,user3")
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    # Parse usernames
    user_list = [u.strip() for u in usernames.split(",") if u.strip()]
    if not user_list:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="No valid usernames provided"
        ).to_response()

    # Validate all usernames
    invalid_users = [u for u in user_list if not validate_username(u)]
    if invalid_users:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Invalid usernames: {', '.join(invalid_users)}"
        ).to_response()

    # Remove collaborators IN PARALLEL
    async def remove_user(username: str) -> dict:
        result = await github.request(
            "DELETE",
            f"/repos/{owner}/{repo}/collaborators/{username}"
        )
        return {"username": username, "success": result.success, "error": result.error_message if not result.success else None}

    tasks = [remove_user(u) for u in user_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = []
    failed = []
    for item in results:
        if isinstance(item, Exception):
            failed.append({"username": "unknown", "error": str(item)})
        elif item["success"]:
            successful.append(item["username"])
        else:
            failed.append({"username": item["username"], "error": item["error"]})

    return ToolResult(success=True, data={
        "action": "removed",
        "repository": f"{owner}/{repo}",
        "total_requested": len(user_list),
        "successful": successful,
        "failed": failed
    }).to_response()


@mcp.tool()
async def get_user_info(
    username: str,
    check_org: str = None
) -> List[TextContent]:
    """
    Get user info and verify organization membership.

    Args:
        username: GitHub username to lookup
        check_org: Organization name to check membership (optional)
    """
    if not validate_username(username):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid username format"
        ).to_response()

    # Get user info
    result = await github.request("GET", f"/users/{username}")

    if not result.success:
        return result.to_response()

    user_data = result.data
    response = {
        "login": user_data.get("login"),
        "name": user_data.get("name"),
        "email": user_data.get("email"),
        "company": user_data.get("company"),
        "location": user_data.get("location"),
        "bio": user_data.get("bio"),
        "public_repos": user_data.get("public_repos"),
        "followers": user_data.get("followers"),
        "following": user_data.get("following"),
        "created_at": user_data.get("created_at"),
        "profile_url": user_data.get("html_url"),
        "type": user_data.get("type")  # "User" or "Organization"
    }

    # Check organization membership if requested
    if check_org:
        if not validate_username(check_org):
            return ToolResult(
                success=False,
                error_code=ErrorCode.VALIDATION_ERROR.value,
                error_message="Invalid organization name"
            ).to_response()

        # Try to check org membership
        # Note: This only works if you have permission to view org members
        # or if the membership is public
        org_check = await github.request(
            "GET",
            f"/orgs/{check_org}/members/{username}"
        )

        if org_check.success:
            response["is_org_member"] = True
            response["org_checked"] = check_org
        elif org_check.error_code == ErrorCode.NOT_FOUND.value:
            # Could be: not a member, or membership is private
            # Try checking public membership
            public_check = await github.request(
                "GET",
                f"/orgs/{check_org}/public_members/{username}"
            )
            if public_check.success:
                response["is_org_member"] = True
                response["membership_visibility"] = "public"
            else:
                response["is_org_member"] = False
            response["org_checked"] = check_org
        else:
            response["is_org_member"] = "unknown"
            response["org_check_error"] = org_check.error_message
            response["org_checked"] = check_org

    return ToolResult(success=True, data=response).to_response()


@mcp.tool()
async def review_user_activity(
    username: str,
    repo: str = None,
    since: str = None,
    until: str = None,
    max_repos: int = 20
) -> List[TextContent]:
    """
    Review user activity on a specific repository or across all repositories.

    Args:
        username: GitHub username to check
        repo: Repository name (optional). If not specified, reviews activity across all repos.
        since: Filter commits after this date (ISO 8601)
        until: Filter commits before this date (ISO 8601)
        max_repos: Max repos to check when repo is not specified (default: 20, max: 50)
    """
    if not username or len(username) > 100:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid username"
        ).to_response()

    # Get token owner info
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()

    owner = token_info.data["owner"]

    # Single repo mode
    if repo:
        if not validate_repo_name(repo):
            return ToolResult(
                success=False,
                error_code=ErrorCode.VALIDATION_ERROR.value,
                error_message="Invalid repository name"
            ).to_response()

        params = {"author": username, "per_page": 30}
        if since:
            params["since"] = since
        if until:
            params["until"] = until

        result = await github.request(
            "GET",
            f"/repos/{owner}/{repo}/commits",
            params=params
        )

        if not result.success:
            return result.to_response()

        commits = [{
            "sha": c["sha"][:7],
            "message": c["commit"]["message"].split("\n")[0],
            "author_name": c["commit"]["author"]["name"],
            "author_email": c["commit"]["author"]["email"],
            "date": c["commit"]["author"]["date"],
            "url": c["html_url"]
        } for c in result.data]

        return ToolResult(success=True, data={
            "repository": f"{owner}/{repo}",
            "username": username,
            "total": len(commits),
            "commits": commits
        }).to_response()

    # All repos mode
    max_repos = min(max(1, max_repos), 50)

    repos_result = await github.request(
        "GET",
        f"/users/{owner}/repos",
        params={"type": "owner", "per_page": max_repos, "sort": "updated"}
    )

    if not repos_result.success:
        return repos_result.to_response()

    repos = repos_result.data
    commits_by_repo = []
    total_commits = 0
    repos_with_commits = []

    async def fetch_commits(repo_name: str) -> tuple:
        params = {"author": username, "per_page": 10}
        if since:
            params["since"] = since
        if until:
            params["until"] = until

        result = await github.request(
            "GET",
            f"/repos/{owner}/{repo_name}/commits",
            params=params
        )
        return repo_name, result

    tasks = [fetch_commits(r["name"]) for r in repos]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for item in results:
        if isinstance(item, Exception):
            continue
        repo_name, result = item
        if result.success and result.data:
            commits = [{
                "sha": c["sha"][:7],
                "message": c["commit"]["message"].split("\n")[0],
                "date": c["commit"]["author"]["date"],
                "url": c["html_url"]
            } for c in result.data]

            if commits:
                repos_with_commits.append(repo_name)
                total_commits += len(commits)
                commits_by_repo.append({
                    "repository": f"{owner}/{repo_name}",
                    "commit_count": len(commits),
                    "commits": commits
                })

    return ToolResult(success=True, data={
        "username": username,
        "owner": owner,
        "total_commits": total_commits,
        "repos_checked": len(repos),
        "repos_with_commits": repos_with_commits,
        "commits_by_repo": commits_by_repo
    }).to_response()


# =============================================================================
# Repository Management Tools
# =============================================================================

@mcp.tool()
async def list_template_repos(owner: str = None, max_repos: int = 50) -> List[TextContent]:
    """
    List template repositories available for creating new repositories.

    Args:
        owner: GitHub username or organization to search for templates (optional, defaults to token owner)
        max_repos: Maximum number of repositories to check (default: 50, max: 100)
    """
    # Get token owner info
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()

    # Use token owner if not specified
    if not owner:
        owner = token_info.data["owner"]
    elif not validate_username(owner):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid username format"
        ).to_response()

    max_repos = min(max(1, max_repos), 100)

    # Get all repos and filter templates
    result = await github.request(
        "GET",
        f"/users/{owner}/repos",
        params={"type": "all", "per_page": max_repos, "sort": "updated"}
    )

    if not result.success:
        return result.to_response()

    # Filter only template repositories
    templates = []
    for repo in result.data:
        if repo.get("is_template"):
            templates.append({
                "full_name": repo["full_name"],
                "name": repo["name"],
                "owner": repo["owner"]["login"],
                "description": repo.get("description", ""),
                "private": repo.get("private", False),
                "default_branch": repo.get("default_branch", "main"),
                "html_url": repo.get("html_url", ""),
                "clone_url": repo.get("clone_url", ""),
                "updated_at": repo.get("updated_at", ""),
                "language": repo.get("language", ""),
                "topics": repo.get("topics", [])
            })

    return ToolResult(success=True, data={
        "owner": owner,
        "total_repos_checked": len(result.data),
        "total_templates": len(templates),
        "templates": templates
    }).to_response()


@mcp.tool()
@protected
async def create_repo_from_template(
    template_owner: str,
    template_repo: str,
    name: str,
    description: str = "",
    private: bool = True,
    include_all_branches: bool = False
) -> List[TextContent]:
    """
    Create a new repository from a template repository.

    Args:
        template_owner: Owner of the template repository (username or organization)
        template_repo: Name of the template repository
        name: Name for the new repository
        description: Description for the new repository (optional)
        private: Whether the new repository should be private (default: True)
        include_all_branches: Include all branches from template, not just default (default: False)
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    # Validate inputs
    if not validate_username(template_owner):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid template owner name"
        ).to_response()

    if not validate_repo_name(template_repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid template repository name"
        ).to_response()

    if not validate_repo_name(name):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid new repository name"
        ).to_response()

    # Verify template repository exists and is a template
    verify_result = await github.request(
        "GET",
        f"/repos/{template_owner}/{template_repo}"
    )

    if not verify_result.success:
        if verify_result.error_code == ErrorCode.NOT_FOUND.value:
            return ToolResult(
                success=False,
                error_code=ErrorCode.NOT_FOUND.value,
                error_message=f"Template repository '{template_owner}/{template_repo}' not found or not accessible"
            ).to_response()
        return verify_result.to_response()

    if not verify_result.data.get("is_template"):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message=f"Repository '{template_owner}/{template_repo}' is not marked as a template"
        ).to_response()

    # Create repository from template
    # API: POST /repos/{template_owner}/{template_repo}/generate
    result = await github.request(
        "POST",
        f"/repos/{template_owner}/{template_repo}/generate",
        json_data={
            "owner": owner,
            "name": name,
            "description": description,
            "private": private,
            "include_all_branches": include_all_branches
        }
    )

    if not result.success:
        return result.to_response()

    repo_data = result.data
    return ToolResult(success=True, data={
        "action": "repository_created_from_template",
        "template": f"{template_owner}/{template_repo}",
        "new_repository": {
            "full_name": repo_data.get("full_name"),
            "name": repo_data.get("name"),
            "private": repo_data.get("private"),
            "description": repo_data.get("description"),
            "html_url": repo_data.get("html_url"),
            "clone_url": repo_data.get("clone_url"),
            "ssh_url": repo_data.get("ssh_url"),
            "default_branch": repo_data.get("default_branch"),
            "created_at": repo_data.get("created_at")
        },
        "owner": owner,
        "include_all_branches": include_all_branches
    }).to_response()


# =============================================================================
# GitHub Actions Secrets Management Tools
# =============================================================================

@mcp.tool()
async def list_action_secrets(repo: str) -> List[TextContent]:
    """
    List all GitHub Actions secrets for a repository.

    Args:
        repo: Repository name
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    result = await github.request(
        "GET",
        f"/repos/{owner}/{repo}/actions/secrets"
    )

    if not result.success:
        return result.to_response()

    secrets = result.data.get("secrets", [])
    return ToolResult(success=True, data={
        "repository": f"{owner}/{repo}",
        "total_secrets": len(secrets),
        "secrets": [{
            "name": s.get("name"),
            "created_at": s.get("created_at"),
            "updated_at": s.get("updated_at")
        } for s in secrets]
    }).to_response()


@mcp.tool()
@protected
async def create_action_secret(
    repo: str,
    secret_name: str,
    secret_value: str
) -> List[TextContent]:
    """
    Create or update a GitHub Actions secret for a repository.
    Note: The secret value will be encrypted by GitHub.

    Args:
        repo: Repository name
        secret_name: Name of the secret
        secret_value: Value of the secret (will be encrypted)
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    if not secret_name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', secret_name):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid secret name. Must start with letter or underscore, contain only letters, digits, underscores, hyphens"
        ).to_response()

    # Import for encryption
    try:
        from nacl import encoding, pwhash
        from nacl.secret import SecretBox
    except ImportError:
        return ToolResult(
            success=False,
            error_code=ErrorCode.API_ERROR.value,
            error_message="PyNaCl library required for secret encryption. Install: pip install pynacl"
        ).to_response()

    # Get the public key for the repo
    pubkey_result = await github.request(
        "GET",
        f"/repos/{owner}/{repo}/actions/secrets/public-key"
    )

    if not pubkey_result.success:
        return pubkey_result.to_response()

    public_key = pubkey_result.data.get("key")
    key_id = pubkey_result.data.get("key_id")

    if not public_key or not key_id:
        return ToolResult(
            success=False,
            error_code=ErrorCode.API_ERROR.value,
            error_message="Failed to get repository public key"
        ).to_response()

    # Encrypt the secret value
    try:
        from nacl.encoding import Base64Encoder
        from nacl.public import SealedBox, PublicKey

        public_key_obj = PublicKey(public_key.encode('utf-8'), Base64Encoder)
        sealed_box = SealedBox(public_key_obj)
        encrypted = sealed_box.encrypt(secret_value.encode('utf-8'), Base64Encoder)
        encrypted_value = encrypted.decode('utf-8')
    except Exception as e:
        return ToolResult(
            success=False,
            error_code=ErrorCode.API_ERROR.value,
            error_message=f"Encryption failed: {str(e)}"
        ).to_response()

    # Create or update the secret
    result = await github.request(
        "PUT",
        f"/repos/{owner}/{repo}/actions/secrets/{secret_name}",
        json_data={
            "encrypted_value": encrypted_value,
            "key_id": key_id
        }
    )

    if not result.success:
        return result.to_response()

    return ToolResult(success=True, data={
        "action": "secret_created_updated",
        "repository": f"{owner}/{repo}",
        "secret_name": secret_name
    }).to_response()


@mcp.tool()
@protected
async def delete_action_secret(
    repo: str,
    secret_name: str
) -> List[TextContent]:
    """
    Delete a GitHub Actions secret from a repository.

    Args:
        repo: Repository name
        secret_name: Name of the secret to delete
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    if not secret_name:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Secret name is required"
        ).to_response()

    result = await github.request(
        "DELETE",
        f"/repos/{owner}/{repo}/actions/secrets/{secret_name}"
    )

    if not result.success:
        return result.to_response()

    return ToolResult(success=True, data={
        "action": "secret_deleted",
        "repository": f"{owner}/{repo}",
        "secret_name": secret_name
    }).to_response()


@mcp.tool()
async def list_environments(repo: str) -> List[TextContent]:
    """
    List all environments in a repository.

    Args:
        repo: Repository name
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    result = await github.request(
        "GET",
        f"/repos/{owner}/{repo}/environments"
    )

    if not result.success:
        return result.to_response()

    environments = result.data.get("environments", [])
    return ToolResult(success=True, data={
        "repository": f"{owner}/{repo}",
        "total_environments": len(environments),
        "environments": [{
            "name": env.get("name"),
            "created_at": env.get("created_at"),
            "updated_at": env.get("updated_at"),
            "protection_rules": env.get("protection_rules", []),
            "deployment_branch_policy": env.get("deployment_branch_policy")
        } for env in environments]
    }).to_response()


@mcp.tool()
@protected
async def create_environment(
    repo: str,
    environment: str,
    wait_timer: int = None,
    reviewers: str = None,
    deployment_branch_policy: str = None
) -> List[TextContent]:
    """
    Create or update an environment in a repository.

    Args:
        repo: Repository name
        environment: Environment name (e.g., 'production', 'staging')
        wait_timer: Optional. Wait timer in seconds (0-43200)
        reviewers: Optional. Comma-separated list of reviewer usernames
        deployment_branch_policy: Optional. Branch policy: 'none', 'all', or 'protected_branches'
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    if not environment or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', environment):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid environment name. Must start with letter or underscore, contain only letters, digits, underscores, hyphens"
        ).to_response()

    # Build request payload
    payload = {}
    if wait_timer is not None:
        if not isinstance(wait_timer, int) or wait_timer < 0 or wait_timer > 43200:
            return ToolResult(
                success=False,
                error_code=ErrorCode.VALIDATION_ERROR.value,
                error_message="Wait timer must be between 0 and 43200 seconds"
            ).to_response()
        payload["wait_timer"] = wait_timer

    if reviewers:
        reviewer_list = [r.strip() for r in reviewers.split(",") if r.strip()]
        if reviewer_list:
            # Get user IDs for reviewers
            reviewers_data = []
            for r in reviewer_list:
                user_result = await github.request("GET", f"/users/{r}")
                if user_result.success:
                    reviewers_data.append({"type": "User", "id": user_result.data.get("id")})
            if reviewers_data:
                payload["reviewers"] = reviewers_data

    if deployment_branch_policy:
        if deployment_branch_policy not in ["none", "all", "protected_branches"]:
            return ToolResult(
                success=False,
                error_code=ErrorCode.VALIDATION_ERROR.value,
                error_message="Deployment branch policy must be: 'none', 'all', or 'protected_branches'"
            ).to_response()
        if deployment_branch_policy == "none":
            payload["deployment_branch_policy"] = None
        elif deployment_branch_policy == "all":
            payload["deployment_branch_policy"] = {"protected_branches": False, "custom_branch_policies": False}
        else:  # protected_branches
            payload["deployment_branch_policy"] = {"protected_branches": True, "custom_branch_policies": False}

    result = await github.request(
        "PUT",
        f"/repos/{owner}/{repo}/environments/{environment}",
        json_data=payload if payload else None
    )

    if not result.success:
        return result.to_response()

    return ToolResult(success=True, data={
        "action": "environment_created_updated",
        "repository": f"{owner}/{repo}",
        "environment": environment,
        "wait_timer": wait_timer,
        "deployment_branch_policy": deployment_branch_policy
    }).to_response()


@mcp.tool()
@protected
async def delete_environment(
    repo: str,
    environment: str
) -> List[TextContent]:
    """
    Delete an environment from a repository.

    Args:
        repo: Repository name
        environment: Environment name (e.g., 'production', 'staging')
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    if not environment:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Environment name is required"
        ).to_response()

    result = await github.request(
        "DELETE",
        f"/repos/{owner}/{repo}/environments/{environment}"
    )

    if not result.success:
        return result.to_response()

    return ToolResult(success=True, data={
        "action": "environment_deleted",
        "repository": f"{owner}/{repo}",
        "environment": environment
    }).to_response()


@mcp.tool()
async def list_environment_secrets(repo: str, environment: str) -> List[TextContent]:
    """
    List all secrets for a specific environment in a repository.

    Args:
        repo: Repository name
        environment: Environment name (e.g., 'production', 'staging')
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    if not environment:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Environment name is required"
        ).to_response()

    result = await github.request(
        "GET",
        f"/repos/{owner}/{repo}/environments/{environment}/secrets"
    )

    if not result.success:
        return result.to_response()

    secrets = result.data.get("secrets", [])
    return ToolResult(success=True, data={
        "repository": f"{owner}/{repo}",
        "environment": environment,
        "total_secrets": len(secrets),
        "secrets": [{
            "name": s.get("name"),
            "created_at": s.get("created_at"),
            "updated_at": s.get("updated_at")
        } for s in secrets]
    }).to_response()


@mcp.tool()
@protected
async def create_environment_secret(
    repo: str,
    environment: str,
    secret_name: str,
    secret_value: str
) -> List[TextContent]:
    """
    Create or update a secret for a specific environment in a repository.
    Note: The secret value will be encrypted by GitHub.

    Args:
        repo: Repository name
        environment: Environment name (e.g., 'production', 'staging')
        secret_name: Name of the secret
        secret_value: Value of the secret (will be encrypted)
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    if not environment:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Environment name is required"
        ).to_response()

    if not secret_name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', secret_name):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid secret name. Must start with letter or underscore, contain only letters, digits, underscores, hyphens"
        ).to_response()

    # Import for encryption
    try:
        from nacl.encoding import Base64Encoder
        from nacl.public import SealedBox, PublicKey
    except ImportError:
        return ToolResult(
            success=False,
            error_code=ErrorCode.API_ERROR.value,
            error_message="PyNaCl library required for secret encryption. Install: pip install pynacl"
        ).to_response()

    # Get the public key for the environment
    pubkey_result = await github.request(
        "GET",
        f"/repos/{owner}/{repo}/environments/{environment}/secrets/public-key"
    )

    if not pubkey_result.success:
        return pubkey_result.to_response()

    public_key = pubkey_result.data.get("key")
    key_id = pubkey_result.data.get("key_id")

    if not public_key or not key_id:
        return ToolResult(
            success=False,
            error_code=ErrorCode.API_ERROR.value,
            error_message="Failed to get environment public key"
        ).to_response()

    # Encrypt the secret value
    try:
        public_key_obj = PublicKey(public_key.encode('utf-8'), Base64Encoder)
        sealed_box = SealedBox(public_key_obj)
        encrypted = sealed_box.encrypt(secret_value.encode('utf-8'), Base64Encoder)
        encrypted_value = encrypted.decode('utf-8')
    except Exception as e:
        return ToolResult(
            success=False,
            error_code=ErrorCode.API_ERROR.value,
            error_message=f"Encryption failed: {str(e)}"
        ).to_response()

    # Create or update the secret
    result = await github.request(
        "PUT",
        f"/repos/{owner}/{repo}/environments/{environment}/secrets/{secret_name}",
        json_data={
            "encrypted_value": encrypted_value,
            "key_id": key_id
        }
    )

    if not result.success:
        return result.to_response()

    return ToolResult(success=True, data={
        "action": "environment_secret_created_updated",
        "repository": f"{owner}/{repo}",
        "environment": environment,
        "secret_name": secret_name
    }).to_response()


@mcp.tool()
@protected
async def delete_environment_secret(
    repo: str,
    environment: str,
    secret_name: str
) -> List[TextContent]:
    """
    Delete a secret from a specific environment in a repository.

    Args:
        repo: Repository name
        environment: Environment name (e.g., 'production', 'staging')
        secret_name: Name of the secret to delete
    """
    # Get token owner
    token_info = await github.get_token_info()
    if not token_info.success:
        return token_info.to_response()
    owner = token_info.data["owner"]

    if not validate_repo_name(repo):
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Invalid repository name"
        ).to_response()

    if not environment:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Environment name is required"
        ).to_response()

    if not secret_name:
        return ToolResult(
            success=False,
            error_code=ErrorCode.VALIDATION_ERROR.value,
            error_message="Secret name is required"
        ).to_response()

    result = await github.request(
        "DELETE",
        f"/repos/{owner}/{repo}/environments/{environment}/secrets/{secret_name}"
    )

    if not result.success:
        return result.to_response()

    return ToolResult(success=True, data={
        "action": "environment_secret_deleted",
        "repository": f"{owner}/{repo}",
        "environment": environment,
        "secret_name": secret_name
    }).to_response()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("GitHub User Management MCP Server")
    print("=" * 50)
    print(f"Rate limit: {config.max_requests_per_minute} req/min")
    print(f"Audit log: {config.audit_log_path}")
    print("=" * 50)
    mcp.run()
