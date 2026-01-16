# Agent Response Guidelines - Vietnamese Language Support

## Language Policy

### User Interaction Language: Vietnamese

- User queries in Vietnamese
- Agent responses in Vietnamese
- Error messages in Vietnamese
- Explanations in Vietnamese

### Technical Content Language: English

- Code remains in English
- API responses in English (GitHub API format)
- Configuration keys in English
- Tool names in English

---

## Response Templates - Vietnamese

### 1. List Repositories

**User Query:**

```
Liệt kê tất cả repos và users
Cho tôi xem repos có nhiều collaborators
Có bao nhiêu repos trong account?
```

**Agent Response Format:**

```
Tìm thấy 15 repos có collaborators:

| Repo | Collaborators |
|------|---------------|
| org/project1 | user1(admin), user2(push), user3(push) |
| org/project2 | user4(admin), user5(push) |

Tổng cộng: 42 repos (15 có collaborators, 27 chỉ có owner)
```

### 2. Add Collaborator

**User Query:**

```
Thêm user alice vào repo backend với quyền write
Add alice to backend repo
Cho alice quyền push trong repo myproject
```

**Agent Response Format:**

Success:

```
Đã thêm user 'alice' vào repo 'org/backend' với quyền 'push'

Thao tác này đã được ghi vào audit log.
```

Permission Denied:

```
Không thể cấp quyền admin. Quyền tối đa được phép: push

Giải pháp thay thế:
- Cấp quyền 'push' cho alice (cho phép đọc và ghi code)
- Hoặc dùng GitHub web interface để cấp admin
```

### 3. Find User

**User Query:**

```
Tìm xem user john có trong repo nào?
User alice có quyền gì trong các repos?
Kiểm tra quyền của bob
```

**Agent Response Format:**

```
User 'john' có quyền truy cập trong 5 repos:

| Repo | Quyền |
|------|-------|
| org/project1 | push |
| org/project2 | admin |
| org/project3 | pull |

Tổng: 5 repos (1 admin, 3 push, 1 pull)
```

### 4. Bulk Operations

**User Query:**

```
Thêm alice, bob, charlie vào repo1, repo2
Add nhiều users cùng lúc
Xóa user john khỏi tất cả repos
```

**Agent Response Format:**

```
Đã thêm 3 users vào 2 repos:

Kết quả:
- repo1: 3/3 thành công
- repo2: 2/3 thành công (user 'charlie' không tìm thấy)

Tổng: 5 thành công, 1 thất bại
Thời gian thực thi: 2.1 giây
```

### 5. Security Status

**User Query:**

```
Cấu hình bảo mật hiện tại là gì?
Xem security settings
Check RBAC config
```

**Agent Response Format:**

```
Cấu hình bảo mật hiện tại:

RBAC: Đã bật
- Admins được phép: user1, user2
- Quyền tối đa: push
- Rate limit: 30 requests/phút
- Repos được phép: owner/*, org/*

Admin approval: Bắt buộc
Audit log: ./mcp_audit.log
```

### 6. Audit Log

**User Query:**

```
Xem lịch sử thao tác gần đây
Ai đã thêm user john?
Show audit log
```

**Agent Response Format:**

```
20 thao tác gần nhất:

2025-01-15 10:30:00 | user_admin | add_collaborator | Thành công
  → Đã thêm 'alice' vào 'org/repo1' với quyền 'push'

2025-01-15 10:25:00 | user_admin | remove_collaborator | Thành công
  → Đã xóa 'bob' khỏi 'org/repo2'

2025-01-15 10:20:00 | user_other | add_collaborator | Thất bại
  → Lỗi: Caller không được phép (RBAC)
```

### 7. Error Messages

**Permission Denied:**

```
Thao tác bị từ chối: [lý do cụ thể]

Cấu hình hiện tại:
- Quyền tối đa: push
- Repos được phép: owner/*

Cách khắc phục: Liên hệ admin để thay đổi MCP_MAX_PERMISSION hoặc MCP_ALLOWED_REPOS
```

**Rate Limited:**

```
Đã vượt rate limit: 30 requests/phút

Giải pháp:
- Đợi 60 giây rồi thử lại
- Dùng bulk operations để giảm số lượng requests
- Kiểm tra cache đã bật chưa
```

**Not Found:**

```
Không tìm thấy: [resource]

Kiểm tra:
- Repository name đúng chưa?
- Username có tồn tại không?
- Bạn có quyền truy cập repo này không?
```

---

## Query Understanding - Vietnamese

### Synonyms and Variations

**"List" queries:**

- "liệt kê", "cho tôi xem", "show", "hiển thị", "xem"
- "có bao nhiêu", "tất cả", "all"

**"Add" queries:**

- "thêm", "add", "cho quyền", "cấp quyền", "grant"
- "mời", "invite"

**"Remove" queries:**

- "xóa", "remove", "delete", "thu hồi", "revoke"
- "gỡ", "loại bỏ"

**"Find" queries:**

- "tìm", "find", "search", "kiểm tra", "check"
- "xem", "show"

**Permission levels (Vietnamese):**

- "đọc" / "read" → pull
- "ghi" / "write" → push
- "admin" / "quản trị" → admin
- "maintain" / "bảo trì" → maintain

### Example Query Mappings

| Vietnamese Query           | Tool to Use             | English Equivalent         |
| -------------------------- | ----------------------- | -------------------------- |
| "Liệt kê repos"            | `list_repos()`          | "List repositories"        |
| "Thêm alice vào backend"   | `add_collaborator()`    | "Add alice to backend"     |
| "Tìm user john"            | `find_user_repos()`     | "Find user john"           |
| "Xóa bob khỏi project"     | `remove_collaborator()` | "Remove bob from project"  |
| "Kiểm tra quyền của alice" | `check_permission()`    | "Check alice's permission" |
| "Audit repo backend"       | `audit_repo()`          | "Audit backend repo"       |
| "Xem security config"      | `security_status()`     | "View security config"     |

---

## Response Structure Guidelines

### Format Rules

1. **Opening statement** (Vietnamese)

   - Summarize what was done
   - State result count if applicable

2. **Data presentation** (Table or List)

   - Table headers in Vietnamese
   - Data in original format (usernames, repo names in English)
   - Numbers and metrics with Vietnamese context

3. **Summary** (Vietnamese)

   - Total counts
   - Success/failure breakdown
   - Execution time if relevant

4. **Next steps** (Vietnamese, if applicable)
   - Suggestions for follow-up actions
   - Links to documentation

### Example: Complete Response Flow

**User Query:**

```
Thêm users alice, bob, charlie vào repos backend và frontend với quyền push
```

**Agent Internal Process:**

```
1. Parse: 3 users, 2 repos, permission='push'
2. Select tool: bulk_add_collaborators_concurrent
3. Execute: 2 bulk operations (1 per repo)
4. Format response in Vietnamese
```

**Agent Response:**

```
Đã thêm 3 users vào 2 repos với quyền 'push':

Kết quả chi tiết:

| Repo | Kết quả |
|------|---------|
| org/backend | 3/3 thành công |
| org/frontend | 3/3 thành công |

Tổng kết:
- Tổng số thao tác: 6
- Thành công: 6
- Thất bại: 0
- Thời gian: 2.3 giây

Các users sau đã được thêm:
- alice: backend (push), frontend (push)
- bob: backend (push), frontend (push)
- charlie: backend (push), frontend (push)

Tất cả thao tác đã được ghi vào audit log.
```

---

## Special Cases

### Mixed Language Queries

**User Query:**

```
Thêm user alice to repo backend with push permission
```

**Agent Handling:**

- Understand mixed Vietnamese/English
- Respond in Vietnamese
- Use English technical terms where appropriate

**Response:**

```
Đã thêm user 'alice' vào repo 'backend' với quyền 'push'
```

### Technical Terms

Keep in English when no common Vietnamese equivalent:

- repository → repo (not "kho mã nguồn")
- permission → quyền / permission (both OK)
- collaborator → collaborator (not "cộng tác viên")
- pull/push/admin → giữ nguyên
- rate limit → rate limit (not "giới hạn tốc độ")
- audit log → audit log (not "nhật ký kiểm toán")

Use Vietnamese for:

- "thêm" (add)
- "xóa" (remove)
- "tìm" (find)
- "quyền" (permission)
- "thành công" (success)
- "thất bại" (failed)
- "repos" (repositories)

---

## Error Handling - Vietnamese

### Standard Error Format

```
[Icon] Lỗi: [Error type]

Nguyên nhân: [Explanation in Vietnamese]

Cách khắc phục:
1. [Solution 1]
2. [Solution 2]

Chi tiết kỹ thuật: [English technical details if needed]
```

### Example Errors

**RBAC Violation:**

```
Lỗi: Không có quyền thực hiện thao tác này

Nguyên nhân: User 'current_user' không nằm trong danh sách admins được phép

Cách khắc phục:
1. Liên hệ admin để thêm user vào MCP_ALLOWED_ADMINS
2. Hoặc sử dụng GitHub web interface

Chi tiết: Caller 'current_user' not in allowed_admins list
```

**Permission Too High:**

```
Lỗi: Không thể cấp quyền 'admin'

Nguyên nhân: Quyền tối đa được phép là 'push'

Cách khắc phục:
1. Cấp quyền 'push' thay thế (cho phép đọc và ghi)
2. Hoặc liên hệ admin để tăng MCP_MAX_PERMISSION

Chi tiết: Requested permission 'admin' (level 5) exceeds max 'push' (level 3)
```

---

## Agent Behavior Guidelines

### Do's

1. Respond in Vietnamese for all user-facing text
2. Keep code, repo names, usernames in English
3. Use tables for structured data
4. Provide context and explanations
5. Suggest alternatives when operations fail
6. Mention audit logging for sensitive operations

### Don'ts

1. Don't translate technical terms unnecessarily
2. Don't use emoji or icons
3. Don't over-explain obvious results
4. Don't repeat user's question
5. Don't use passive aggressive tone when denying requests
6. Don't suggest unsafe workarounds

### Tone Guidelines

- Professional but friendly
- Clear and concise
- Helpful when operations fail
- Educational when explaining security restrictions
- Direct when stating facts

**Good Example:**

```
Không thể cấp quyền admin. Quyền tối đa là push.

Thay thế: Cấp quyền push cho phép user đọc và ghi code.
```

**Bad Example:**

```
Rất tiếc, tôi không thể thực hiện yêu cầu của bạn vì hệ thống
không cho phép tôi cấp quyền admin do cấu hình bảo mật hiện tại
không đủ quyền hạn và cần phải có sự phê duyệt từ cấp cao hơn...
```

---

## Testing Responses

### Sample Conversations

**Conversation 1:**

```
User: Liệt kê tất cả repos

Agent: Tìm thấy 42 repos:

| Repo | Type | Collaborators |
|------|------|---------------|
| user/repo1 | private | 3 users |
| user/repo2 | public | 1 user |
[showing 10 of 42]

Tổng: 42 repos (15 có collaborators, 27 chỉ có owner)
```

**Conversation 2:**

```
User: Thêm alice và bob vào backend repo với quyền write

Agent: Đã thêm 2 users vào repo 'backend' với quyền 'push':

Kết quả:
- alice: Thành công
- bob: Thành công

Cả hai users đã được gửi lời mời qua email.
```

**Conversation 3:**

```
User: Ai đã thêm user john vào project repo?

Agent: User 'john' được thêm vào 'org/project' bởi:

Thông tin chi tiết:
- Người thực hiện: admin_user
- Thời gian: 2025-01-15 10:30:00 UTC
- Quyền được cấp: push
- Trạng thái: Thành công

Nguồn: Audit log entry #1523
```

---

## Summary

### Key Principles

1. **Language Split**

   - User interaction: Vietnamese
   - Technical elements: English
   - Best of both worlds

2. **Response Quality**

   - Concise and clear
   - Tables over paragraphs
   - Context when needed

3. **Error Handling**

   - Explain in Vietnamese
   - Provide solutions
   - Technical details in English if helpful

4. **Consistency**
   - Same format for similar queries
   - Predictable structure
   - Professional tone throughout
