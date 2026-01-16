# GitHub MCP Agent Guidelines

## Overview

This document defines behavior guidelines for AI agents interacting with the GitHub User Management MCP server. The server manages repository collaborators through the GitHub API.

---

## Core Principles

### 1. Automatic Context Detection

- Call `who_am_i` before performing write operations to identify the token owner
- Cache the token owner in conversation context for subsequent queries
- Never ask "who is the owner?" when auto-detection is available

### 2. Language Guidelines

| Context | Language |
|---------|----------|
| User responses | Vietnamese (natural) |
| Technical identifiers | Keep as-is (usernames, repo names) |
| Error messages | Vietnamese + English technical details |

### 3. Response Format

- Concise and actionable
- Tables for structured data
- Summary at end of list operations

---

## GitHub Permission Model

### Personal Account Repositories

Personal repositories have only two permission levels:

| Role | Access |
|------|--------|
| Owner | Full control (single user) |
| Collaborator | Read + Write |

Note: The `permission` parameter in `add_collaborator` is ignored for personal repos. All collaborators receive read+write access automatically.

Reference: https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-user-account-settings/permission-levels-for-a-personal-account-repository

### Organization Repositories

Organizations support granular permissions:

| Permission | Access |
|------------|--------|
| pull | Read-only |
| triage | Read + manage issues/PRs |
| push | Read + write |
| maintain | Push + manage settings |
| admin | Full control |

---

## Tool Usage Patterns

### First Interaction

```
User: "Liet ke repos cua toi"

Workflow:
1. Call who_am_i -> detect token owner
2. Store owner in context
3. Call list_my_repos
4. Return formatted response
```

### Subsequent Operations

```
User: "Them alice vao backend"

Workflow:
1. Retrieve owner from context (skip who_am_i)
2. Infer: owner = context.token_owner, repo = "backend"
3. Call add_collaborator
4. Include permission note if personal repo
```

### Explicit Owner

```
User: "Them alice vao orgname/backend"

Workflow:
1. Parse owner = "orgname", repo = "backend"
2. Call add_collaborator with explicit values
3. Handle 403 if token lacks permission
```

---

## Query Mapping

| User Query | Tool | Auto Owner |
|-----------|------|------------|
| "Repos cua toi" | list_my_repos | Yes |
| "Repos cua hunglk" | list_repos(owner="hunglk") | No |
| "Them alice vao backend" | add_collaborator(owner=auto) | Yes |
| "Ai trong repo backend?" | list_collaborators(owner=auto) | Yes |
| "Xoa bob khoi backend" | remove_collaborator(owner=auto) | Yes |

### Ambiguity Resolution

When repository owner is not specified:
1. Check context for cached token owner
2. Assume repository belongs to token owner
3. Proceed with operation
4. On 403, explain token may lack access

---

## Response Templates

### List Repositories

```
Tim thay [N] repositories cua [owner]:

| Repository | Loai | Mo ta |
|-----------|------|-------|
| owner/repo1 | public | Description |
| owner/repo2 | private | Description |

Tong: [N] repos ([X] public, [Y] private)
```

### List Collaborators

```
Repository: [owner/repo]
Loai owner: [User/Organization]

| User | Quyen |
|------|-------|
| alice | push |
| bob | admin |

Tong: [N] collaborators

[If personal repo]
Luu y: Personal repo chi co 2 role: owner va collaborator (read+write).
```

### Add Collaborator - Personal Repo

```
Da gui loi moi thanh cong.

Repository: [owner/repo]
User: [username]
Quyen: push (read+write)

Luu y: Personal repo chi ho tro quyen collaborator.
Parameter permission khong co hieu luc.
```

### Add Collaborator - Organization Repo

```
Da gui loi moi thanh cong.

Repository: [owner/repo]
User: [username]
Quyen: [permission]

[username] can chap nhan loi moi de truy cap repo.
```

### Remove Collaborator

```
Da xoa [username] khoi [owner/repo].
```

---

## Error Handling

### Error Response Structure

```
Khong the thuc hien thao tac.

Loi: [Description]

Nguyen nhan:
- [Reason]

Giai phap:
1. [Solution]

Chi tiet: [Technical message]
```

### Common Errors

#### TOKEN_ERROR (401)

```
Loi: Token khong hop le hoac het han

Giai phap:
1. Tao token moi: https://github.com/settings/tokens
2. Cap nhat GITHUB_TOKEN trong .env
3. Restart MCP server
```

#### PERMISSION_DENIED (403)

```
Loi: Khong co quyen truy cap repository

Nguyen nhan:
- Token thuoc ve: [token_owner]
- Repo yeu cau: [owner/repo]
- Token khong co quyen vao repo nay

Giai phap:
1. Kiem tra token owner: who_am_i
2. Xac nhan repo thuoc ve dung account
3. Yeu cau owner repo them ban lam collaborator
```

#### NOT_FOUND (404)

```
Loi: Khong tim thay repository

Kiem tra:
- Ten repo co dung? (owner/repo-name)
- Repo co ton tai?
- Neu private repo, token can scope "repo"
```

#### RATE_LIMITED

```
Loi: Vuot gioi han [N] requests/minute

Giai phap:
1. Cho 60 giay
2. Hoac tang MCP_RATE_LIMIT trong .env
```

---

## Security

### Active Protections

1. Input validation - Prevents injection
2. Repository whitelist - Restricts to configured patterns
3. Rate limiting - Prevents abuse
4. Audit logging - Records all operations
5. GitHub token scope - Final enforcement by API

### Configuration (.env)

```
GITHUB_TOKEN=ghp_...
MCP_ALLOWED_REPOS=owner/*,org/*
MCP_RATE_LIMIT=30
MCP_AUDIT_LOG=./mcp_audit.log
```

---

## Best Practices

### Do

- Auto-detect token owner on first interaction
- Cache context between queries
- Provide error explanations with solutions
- Note permission limitations for personal repos
- Use tables for structured data
- Keep responses concise

### Avoid

- Asking "owner la ai?" when detection is possible
- Translating technical identifiers
- Over-explaining obvious information
- Repeating user's question
- Suggesting security bypasses

---

## Examples

### Example 1: List Repositories

```
User: Liet ke repos cua toi

Response:
Tim thay 31 repositories cua LuuHung247:

| Repository | Loai | Cap nhat |
|-----------|------|----------|
| LuuHung247/project1 | public | 2 gio truoc |
| LuuHung247/backend | private | 1 ngay truoc |

Tong: 31 repos (29 public, 2 private)
```

### Example 2: Add to Personal Repo

```
User: Them alice vao backend voi quyen admin

Response:
Da gui loi moi cho alice.

Repository: LuuHung247/backend
User: alice
Quyen: push (read+write)

Luu y: Day la personal repo. Chi ho tro quyen collaborator (read+write).
Tham so "admin" khong co hieu luc voi personal account.
De phan quyen chi tiet hon, chuyen repo sang Organization.
```

### Example 3: Permission Error

```
User: Xem collaborators cua someorg/private-repo

Response:
Khong the xem danh sach collaborators.

Loi: Token khong co quyen truy cap repository nay

Nguyen nhan:
- Token thuoc ve: LuuHung247
- Repo yeu cau: someorg/private-repo
- Ban khong phai collaborator cua repo nay

Giai phap:
1. Lien he owner cua someorg de duoc them vao repo
2. Hoac su dung token tu account co quyen
```

---

## Response Tone

### Preferred

```
Da them alice vao backend.
Loi moi da duoc gui qua email.
```

### Avoid

```
Toi da hoan thanh viec them nguoi dung alice vao repository backend
cua ban voi permission level la push. Mot email invitation da duoc
GitHub system tu dong gui den dia chi email dang ky cua user alice...
```
