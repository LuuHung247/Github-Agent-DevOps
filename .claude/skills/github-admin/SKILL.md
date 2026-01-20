---
name: github-admin
description: "Quản lý user truy cập vào GitHub repositories thông qua MCP github-admin server."
---

# GitHub User Management

## Overview

Quản lý quyền truy cập GitHub via MCP github-admin. 3 workflows: User Provisioning (thêm user), User Deprovisioning (xóa user), Periodic Audit (rà soát).

**Luôn response bằng tiếng Việt khi user hỏi tiếng Việt.**

## Start with Context

Bắt đầu với `mcp__github-admin__who_am_i` để xác định permissions.

## Workflow 1: User Provisioning (Thêm User)

**Bước 1:** List collaborators → Nếu tìm thấy user, stop.
**Bước 2:** Check pending invitations → Nếu có, hỏi user có muốn resend không.
**Bước 3:** Nếu chưa có, dùng `add_collaborator` (1 user) hoặc `batch_add_collaborators` (nhiều users).
**Bước 4:** Report kết quả.

```
Tools: list_collaborators, list_pending_invitations, cancel_invitation, add_collaborator, batch_add_collaborators
```

## Workflow 2: User Deprovisioning (Xóa User)

**Bước 1:** Check pending invitations → Cancel nếu có.
**Bước 2:** Dùng `remove_collaborator` (1 user) hoặc `batch_remove_collaborators` (nhiều users).
**Bước 3:** Verify removal via `list_collaborators`.
**Bước 4:** Report kết quả.

```
Tools: list_pending_invitations, cancel_invitation, remove_collaborator, batch_remove_collaborators
```

## Workflow 3: Periodic Audit (Rà Soát)

**Bước 1:** Dùng `quick_overview` hoặc `list_all_collaborators` để lấy tổng quan.
**Bước 2:** Với mỗi user đáng ngờ, dùng `review_user_activity` để check hoạt động.
**Bước 3:** Report inactive users (3+ tháng không commit).
**Bước 4:** Nếu user đồng ý, xóa users inactive + cancel old pending invitations.

```
Tools: quick_overview, list_all_collaborators, review_user_activity, remove_collaborator, cancel_invitation
```

## Best Practices

**Batch operations:** Dùng `batch_add_collaborators`, `batch_remove_collaborators` thay vì loop

**Verify before action:**

- Add: `get_user_info` để verify username
- Remove: `review_user_activity` để check còn hoạt động không
- After: `audit_log_recent` để verify operation

**Response style:** Luôn bằng tiếng Việt

## Error Handling

| Error               | Fix                                                       |
| ------------------- | --------------------------------------------------------- |
| `PERMISSION_DENIED` | Check admin rights, token scopes, repo whitelist          |
| `NOT_FOUND`         | Verify username/repo, check typo via `get_user_info`      |
| `RATE_LIMITED`      | Wait, reduce `max_repos`, check rate limit via `who_am_i` |

## Tools Reference

| Tool                           | Purpose                      |
| ------------------------------ | ---------------------------- |
| `who_am_i`                     | Current account info         |
| `quick_overview`               | Overview all repos           |
| `list_collaborators`           | Collaborators of 1 repo      |
| `list_all_collaborators`       | Collaborators of all repos   |
| `list_pending_invitations`     | Pending invites of 1 repo    |
| `list_all_pending_invitations` | Pending invites of all repos |
| `get_user_info`                | Verify user exists           |
| `add_collaborator`             | Add 1 user                   |
| `batch_add_collaborators`      | Add multiple users           |
| `remove_collaborator`          | Remove 1 user                |
| `batch_remove_collaborators`   | Remove multiple users        |
| `cancel_invitation`            | Cancel pending invite        |
| `review_user_activity`         | Check user commits           |
| `audit_log_recent`             | Recent operations            |
| `security_status`              | Security config              |
