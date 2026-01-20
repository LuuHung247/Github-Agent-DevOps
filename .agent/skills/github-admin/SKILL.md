---
name: github-admin
description: "Quản lý user truy cập vào GitHub repositories thông qua MCP github-admin server. Dùng khi: (1) Thêm/mời user vào repo, (2) Xóa user khỏi repo, (3) Rà soát/audit quyền truy cập, (4) Kiểm tra pending invitations, (5) Xem hoạt động commit của user."
---

# GitHub User Management

Quản lý quyền truy cập repo via MCP github-admin.

**LUÔN trả lời bằng tiếng Việt. PHẢI tuân thủ TỪNG BƯỚC trong workflow - KHÔNG được bỏ qua bước nào.**

## Quick Start

`mcp__github-admin__quick_overview` → Lấy thông tin account + collaborators + invitations.

## Workflows

### Thêm User

**PHẢI thực hiện đủ 3 bước theo thứ tự:**

1. `list_collaborators(repo="X")` → Nếu user đã có trong danh sách → DỪNG, báo đã tồn tại
2. `list_pending_invitations(repo="X")` → Nếu user có pending invitation → HỎI user có muốn gửi lại không
3. `add_collaborators(repo, usernames)` → Gửi invitation

### Xóa User

**PHẢI thực hiện đủ 4 bước theo thứ tự:**

1. `list_pending_invitations(repo="X")` → Lấy danh sách pending
2. **NẾU có pending invitation cho user cần xóa** → `cancel_invitation(repo, invitation_id)` để hủy TỪNG invitation
3. `remove_collaborators(repo, usernames)` → Xóa collaborators
4. `list_collaborators(repo="X")` → Verify đã xóa thành công

### Rà Soát Quyền Truy Cập

**PHẢI thực hiện đủ 4 bước:**

1. `quick_overview` hoặc `list_collaborators()` → Lấy danh sách tất cả collaborators
2. `review_user_activity(username)` → Check commits của TỪNG user đáng ngờ
3. Báo cáo users không hoạt động (3+ tháng không commit)
4. **CHỈ xóa khi user xác nhận đồng ý**

## Key Patterns

**Single vs All repos:** `list_collaborators`, `list_pending_invitations`, `review_user_activity` có optional `repo`:
- Có `repo`: chỉ 1 repo
- Không có `repo`: scan tất cả repos

**Verify trước khi action:**
- Thêm: `get_user_info` để verify username tồn tại
- Xóa: `review_user_activity` để check còn hoạt động không

## Error Handling

| Error | Cách xử lý |
|-------|------------|
| `PERMISSION_DENIED` | Kiểm tra admin rights, token scopes |
| `NOT_FOUND` | Verify username/repo qua `get_user_info` |
| `RATE_LIMITED` | Giảm `max_repos` parameter |
