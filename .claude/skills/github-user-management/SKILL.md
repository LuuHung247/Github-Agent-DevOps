---
name: github-user-management
description: Manage GitHub repository collaborators and access permissions. Add/remove users, review pending invitations, audit user activity, and conduct periodic access reviews. Use when managing GitHub repository access, adding collaborators, removing users, checking permissions, or conducting security audits.
compatibility: Requires GITHUB_TOKEN environment variable with repo scope
---

# GitHub User Management

Manage repository collaborators through add/remove operations, invitation management, activity reviews, and security audits.

## Quick Start

**Common tasks:**

- Add collaborator: `github_get_user_info` → `github_add_collaborator`
- Remove access: `github_remove_collaborator`
- Review pending: `github_list_invitations`
- Security audit: `github_list_all_collaborators`

**All tools support:**

- `response_format`: `"json"` (structured data) or `"markdown"` (human-readable)

## Core Workflows

### Add New Collaborator

1. **Verify user exists**

   ```
   github_get_user_info(username="alice", check_org="myorg")
   ```

   Returns: Profile info, org membership status

2. **Check contribution history** (optional but recommended)

   ```
   github_review_user_activity(username="alice")
   ```

   Returns: Recent commits across repositories

3. **Send invitation**

   ```
   github_add_collaborator(owner="myorg", repo="project", username="alice", permission="push")
   ```

   Returns: `invitation_id` (save for potential cancellation), `status` ("pending" or "active")

4. **User accepts** → Becomes active collaborator

**Important:**

- Personal repos: Always grant "push" (read+write), `permission` parameter ignored
- Organization repos: Supports `pull`, `triage`, `push`, `maintain`, `admin`
- Invitations expire after 7 days if not accepted

### Remove Collaborator

1. **Review activity first** (recommended)

   ```
   github_review_user_repo_activity(owner="myorg", repo="project", username="bob")
   ```

2. **Remove access**
   ```
   github_remove_collaborator(owner="myorg", repo="project", username="bob")
   ```
   Effect: Immediate access revocation

### Manage Pending Invitations

**Check pending invitations:**

```
github_list_invitations(owner="myorg", repo="project")
```

Returns: List with `invitation_id`, username, permissions, invited_at

**Cancel invitation:**

```
github_cancel_invitation(owner="myorg", repo="project", invitation_id=12345)
```

**See everything (active + pending):**

```
github_list_collaborators(owner="myorg", repo="project", include_pending=true)
```

### Periodic Access Review

**Monthly/Quarterly audit:**

```
github_list_all_collaborators(max_repos=50)
```

Returns: All collaborators grouped by access count

**For each user:**

```
github_review_user_activity(username="user", max_repos=20)
```

Decision: Remove if inactive or access no longer needed

## Tool Reference

### Discovery Tools

**`github_who_am_i`**

- Get authenticated account info (username, scopes, repo counts)
- Use at session start to confirm identity

**`github_list_my_repos`**

- List current account's repositories
- Parameters: `type` (owner/member/all), `limit`, `offset`
- Returns: Paginated list with `has_more`, `next_offset`

**`github_list_repos`**

- List repositories of any user/organization
- Parameters: `owner`, `type`, `limit`, `offset`

**`github_get_user_info`**

- Get user profile: name, email, bio, public_repos, followers
- Optional: `check_org` to verify organization membership
- **Always use before adding collaborators**

### Collaborator Tools

**`github_list_collaborators`**

- List collaborators for one repository
- Set `include_pending=true` to see pending invitations
- Returns: Permissions (pull/triage/push/maintain/admin)

**`github_list_all_collaborators`**

- Security audit: All collaborators across multiple repos
- Parameters: `max_repos` (default: 20, max: 50)
- Groups users by repository access count

**`github_add_collaborator`** [PROTECTED]

- Send invitation to user (not immediate access)
- Personal repos: Always "push" access
- Organization repos: Specify `permission` level
- Returns: `invitation_id`, `status` (pending/active)

**`github_remove_collaborator`** [PROTECTED]

- Immediately revoke repository access
- Requires: `owner`, `repo`, `username`

### Invitation Tools

**`github_list_invitations`**

- Show pending invitations that haven't been accepted
- Returns: `invitation_id`, invitee, permissions, invited_at, inviter

**`github_cancel_invitation`** [PROTECTED]

- Cancel pending invitation by ID
- Requires: `owner`, `repo`, `invitation_id`

### Activity Review Tools

**`github_review_user_repo_activity`**

- Review commits by user in one repository
- Optional date filters: `since`, `until` (ISO 8601)
- Returns: Commit SHA, message, author, date, URL

**`github_review_user_activity`**

- Check user commits across ALL repositories
- Parameters: `username`, `max_repos`, `since`, `until`
- Returns: Total commits, repos with activity, commits by repo

### Security Tools

**`github_security_status`**

- View security configuration (whitelist, rate limits, audit log)
- Lists active protections

**`github_audit_log_recent`**

- View recent operations history
- Parameters: `limit` (default: 20)
- Shows: timestamp, caller, action, success, details

## Permission Models

### Personal Repositories

- **Only 2 levels:** Owner (full control) or Collaborator (push)
- All collaborators get read+write access
- `permission` parameter is ignored

### Organization Repositories

- **Granular permissions:**
  - `pull`: Read-only
  - `triage`: Read + manage issues/PRs (no code write)
  - `push`: Read + write code
  - `maintain`: Push + manage settings (no admin)
  - `admin`: Full control

## Invitation State Flow

```
github_add_collaborator()
    ↓
status: "pending", invitation_id: 12345
    ↓
┌─────────────────┬──────────────────┐
│  User Accepts   │  User Ignores    │
↓                 ↓                  ↓
status: "active"  Stays "pending"   Expires (7 days)
(collaborator)    (invitation)
```

**Check invitation:**

- Use `github_list_collaborators(include_pending=true)` for full picture
- Use `github_list_invitations` for pending-only view

## Security Features

**Protected operations** (`github_add_collaborator`, `github_remove_collaborator`, `github_cancel_invitation`) enforce:

1. **Repository Whitelist**
   - Operations only on whitelisted repos
   - Configure: `MCP_ALLOWED_REPOS` env variable
   - Patterns: `owner/repo`, `owner/*`, `*/*`

2. **Rate Limiting**
   - Max requests per minute (default: 30)
   - Configure: `MCP_RATE_LIMIT` env variable

3. **Audit Logging**
   - All operations recorded with timestamp, caller, action, success
   - View with `github_audit_log_recent`

4. **Input Validation**
   - Pydantic models prevent injection attacks
   - Username: `^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$`
   - Repo: `^[a-zA-Z0-9._-]+$`

## Error Handling

**Error codes:**

- `PERMISSION_DENIED`: Whitelist or access violation
- `RATE_LIMITED`: Too many requests
- `NOT_FOUND`: User/repo doesn't exist
- `VALIDATION_ERROR`: Invalid input format
- `API_ERROR`: GitHub API error
- `NETWORK_ERROR`: Connection issue
- `TOKEN_ERROR`: Invalid GITHUB_TOKEN

**Example error response:**

```json
{
  "error": true,
  "code": "PERMISSION_DENIED",
  "message": "Repository not in whitelist: owner/repo"
}
```

## Best Practices

### Before Adding Collaborators

- ✅ Verify user exists: `github_get_user_info`
- ✅ Check organization membership if applicable
- ✅ Review contribution history: `github_review_user_activity`
- ✅ Use least privilege: Choose minimal necessary permission

### Periodic Reviews

- 📅 **Monthly:** `github_list_all_collaborators` → Review active users
- 📅 **Weekly:** `github_list_invitations` → Clean up pending invitations
- 📅 **Quarterly:** Full audit with `github_audit_log_recent`

### Invitation Management

- ⏱️ Invitations expire after 7 days - cancel if not accepted
- 💾 Save `invitation_id` from `github_add_collaborator` response
- 🔍 Use `include_pending=true` to see complete access picture

### Response Format

- Use `response_format: "json"` for automation/scripts
- Use `response_format: "markdown"` for human review/reports
- Keep output concise - don't list empty results

## Environment Setup

**Required:**

```bash
GITHUB_TOKEN=ghp_xxxxx  # GitHub personal access token with 'repo' scope
```

**Optional:**

```bash
MCP_ALLOWED_REPOS=owner/*,org/specific-repo  # Whitelist (default: */*)
MCP_RATE_LIMIT=30                             # Requests/minute (default: 30)
MCP_AUDIT_LOG=./logs/mcp_audit.log           # Log path
```

## Common Patterns

**Bulk add collaborators:**

```
For each user in list:
  1. github_get_user_info(username)
  2. github_add_collaborator(owner, repo, username, permission)
  3. Save invitation_id for tracking
```

**Remove inactive users:**

```
1. github_list_all_collaborators()
2. For each user:
   - github_review_user_activity(username, since="2024-01-01")
   - If no activity: github_remove_collaborator()
```

**Clean up pending invitations:**

```
1. github_list_invitations(owner, repo)
2. For each invitation older than 3 days:
   - github_cancel_invitation(invitation_id)
```

**Security audit:**

```
1. github_list_all_collaborators(max_repos=50)
2. For each unexpected user:
   - github_get_user_info(username, check_org="myorg")
   - github_review_user_activity(username)
   - Decision: Keep or remove
3. github_audit_log_recent(limit=100) - Review recent changes
```

## Pagination

List tools support pagination:

- Set `limit` (max results per request)
- Use `offset` to skip results
- Check `has_more` (boolean) for more data
- Use `next_offset` for next page

**Example:**

```
# First page
github_list_my_repos(limit=20, offset=0)
→ has_more: true, next_offset: 20

# Second page
github_list_my_repos(limit=20, offset=20)
→ has_more: false, next_offset: null
```

## Quick Reference

| Task                | Tool                                              |
| ------------------- | ------------------------------------------------- |
| Check my identity   | `github_who_am_i`                                 |
| Find my repos       | `github_list_my_repos`                            |
| Verify user exists  | `github_get_user_info`                            |
| Add collaborator    | `github_add_collaborator`                         |
| See all access      | `github_list_collaborators(include_pending=true)` |
| Remove access       | `github_remove_collaborator`                      |
| Cancel invitation   | `github_cancel_invitation`                        |
| Check contributions | `github_review_user_activity`                     |
| Security audit      | `github_list_all_collaborators`                   |
| View recent changes | `github_audit_log_recent`                         |
