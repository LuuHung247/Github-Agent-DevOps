# CLAUDE.md

File này hướng dẫn Claude Code (claude.ai/code) cách làm việc hiệu quả với code trong repository này.

## Tổng quan Dự án

Đây là MCP (Model Context Protocol) server cung cấp khả năng quản lý GitHub repositories và users. Server expose các tools để quản lý collaborators, permissions, và thực hiện bulk operations trên nhiều repositories.

Server được xây dựng bằng FastMCP và giao tiếp với GitHub REST API v3.

## Cấu hình Môi trường

Environment variables bắt buộc (xem [.env.example](.env.example)):

- `MY_GITHUB_TOKEN`: GitHub Personal Access Token
  - Tạo tại: https://github.com/settings/tokens
  - Scopes bắt buộc: `repo`, `admin:org`, `read:user`

Copy `.env.example` sang `.env` và thêm token:

```bash
cp .env.example .env
# Chỉnh sửa .env và thêm GitHub token của bạn
```

## Chạy Dự án

### Chế độ MCP Server (Production)

Server được cấu hình trong `.mcp.json` và chạy qua uv:

```bash
uv run mcp_admin.py
```

Lệnh này khởi động FastMCP server có thể kết nối từ MCP clients (như Claude Code).

### Chế độ Test

Chạy với flag `--test` để kiểm tra kết nối GitHub API:

```bash
uv run mcp_admin.py --test
```

Lệnh này sẽ liệt kê 3 repositories đầu tiên và collaborators của chúng.

### Entry Point

File `main.py` là placeholder và không được sử dụng bởi MCP server.

## Kiến trúc

### Cấu trúc Cốt lõi

- **[mcp_admin.py](mcp_admin.py)**: MCP server đơn file chứa tất cả tools
- **FastMCP Framework**: Sử dụng decorator `@mcp.tool()` để expose async functions thành MCP tools
- **GitHub API Client**: Async httpx client với error handling qua wrapper `github_request()`

### Nhóm Tools

**Repository Discovery:**

- `list_all_repos()`: Liệt kê tất cả repos mà authenticated user có quyền truy cập
- `list_repo_collaborators()`: Liệt kê collaborators của một repo cụ thể

**User Management:**

- `add_user_to_repo()`: Thêm user với permissions cụ thể (pull/push/admin/maintain/triage)
- `remove_user_from_repo()`: Xóa user khỏi repo
- `update_user_permission()`: Thay đổi permission level của user
- `check_user_permission()`: Kiểm tra access level của user cụ thể

**Bulk Operations:**

- `bulk_add_users()`: Thêm nhiều users vào repo (có rate limiting)
- `bulk_remove_users()`: Xóa nhiều users khỏi repo
- `sync_users_across_repos()`: Copy tất cả collaborators từ source repo sang target repos

**Auditing:**

- `find_user_across_repos()`: Tìm repos mà một user cụ thể có quyền truy cập
- `audit_all_permissions()`: Tạo báo cáo đầy đủ về permissions trên tất cả repos

### Chi tiết Kỹ thuật

**Rate Limiting:**

- Tích hợp `asyncio.sleep()` delays trong bulk operations (0.3-1s giữa các requests)
- Ngăn chặn hitting GitHub API rate limits

**Error Handling:**

- Tất cả tools sử dụng try/except với `httpx.HTTPStatusError`
- Trả về structured dicts với `success` boolean và error messages
- 404 responses được xử lý riêng cho permission checks

**Permissions Model:**
GitHub permission levels (từ thấp đến cao):

- `pull`: Read-only access
- `triage`: Read + triage issues/PRs
- `push`: Read + write access
- `maintain`: Push + manage issues/PRs
- `admin`: Full admin access

## Lưu ý Development

- Python 3.12 bắt buộc (xem [.python-version](.python-version))
- Sử dụng `uv` cho dependency management
- Tất cả API calls đều async dùng httpx
- GitHub API v3 (REST) với token authentication trong headers

---

## 🤖 Hướng dẫn cho Claude Agent

### Nguyên tắc Làm việc

**1. Tool Selection:**

- Dùng MCP tools cho **tất cả** thao tác liên quan đến permissions và users
- **KHÔNG BAO GIỜ** sử dụng GitHub CLI (`gh`) cho user/permission management
- MCP tools đã tối ưu rate limiting và error handling

**2. Response Format - QUAN TRỌNG:**

**✅ LUÔN LUÔN** trả lời theo format sau:

#### Khi list repos/users:

```
| Repo | Users |
|------|-------|
| owner/repo1 | user1(admin), user2(write) |
| owner/repo2 | user3(admin) |
```

#### Khi audit permissions:

```
Tổng: X repos, Y repos có collaborators

| Repo | Collaborators |
|------|---------------|
| repo1 | user1(admin), user2(write), user3(write) |
| repo2 | user4(admin), user5(write) |
```

#### Khi add/remove user:

```
✅ Đã thêm [user] vào [repo] với quyền [permission]
```

hoặc

```
❌ Lỗi: [error message]
```

#### Khi tìm user:

```
User "[username]" có trong X repos:
- repo1 (admin)
- repo2 (write)
- repo3 (write)
```

**❌ TUYỆT ĐỐI KHÔNG:**

- Viết narrative dài dòng giải thích data
- Tạo sections kiểu "Báo cáo đầy đủ", "Thống kê", "Tổng kết"
- Lặp lại thông tin đã rõ ràng trong data
- Dùng emoji quá nhiều (chỉ dùng ✅❌🔒)
- Giải thích lại câu hỏi của user

**3. Data Presentation:**

- Nếu ≤10 items: show tất cả
- Nếu >10 items: show 10 items đầu + "... và X items nữa"
- Repos chỉ có 1 user (owner): nhóm riêng hoặc bỏ qua nếu không được hỏi
- Luôn dùng tables cho structured data

### Workflows Phổ biến

#### Workflow 1: List và Audit

```
User: "List tất cả repos và users"
Agent: [Gọi audit_all_permissions()]
Response:
| Repo | Users |
|------|-------|
[table data]
```

#### Workflow 2: Thêm User

```
User: "Thêm alice vào repo backend với quyền push"
Agent: [Gọi add_user_to_repo(owner="...", repo="backend", username="alice", permission="push")]
Response: ✅ Đã thêm alice vào owner/backend với quyền push
```

#### Workflow 3: Bulk Operations

```
User: "Thêm [user1, user2] vào [repo1, repo2] với quyền write"
Agent: [Gọi bulk_add_users() cho mỗi repo]
Response:
✅ repo1: Đã thêm 2 users
✅ repo2: Đã thêm 2 users
```

#### Workflow 4: Audit và Tìm kiếm

```
User: "Tìm user LonelyLemon trong các repos"
Agent: [Gọi find_user_across_repos(username="LonelyLemon")]
Response:
User "LonelyLemon" có trong 5 repos:
- EduConnect-Backend (write)
- EduConnect-Helm (write)
- EduConnect-transcript (write)
- INT3505E_02_demo (write)
- PaaS_AWS-Education-Web-Frontend (write)
```

### Token Usage Optimization

**Chiến lược tối ưu tokens:**

**1. Phân tích Query trước:**

```
Input: "List repos và users"
→ Identify: Cần tool "audit_all_permissions"
→ Expected output: Table format
→ No need for: Narrative, statistics, explanations
```

**2. Structured Thinking (không output ra user):**

```
Internal:
- Tool: audit_all_permissions()
- Filter: repos with >1 user only
- Format: Markdown table
- Max rows: 10 (nếu >10 thì truncate)

Output:
[Chỉ table, không thêm gì khác]
```

**3. Response Template (dùng trong đầu, không nói ra):**

```
Query type: [list/audit/add/remove/find]
Tool: [tool_name]
Output format: [table/list/single_line]
Data filtering: [what to show/hide]
```

**4. Caching Strategy:**

- Nếu user hỏi liên tiếp về cùng repos → dùng data đã có, không gọi lại API
- Chỉ gọi MCP tool khi:
  - Data chưa có
  - Data đã cũ (>5 phút)
  - User yêu cầu refresh

**5. Selective Data Return:**

```python
# Ví dụ internal logic:
if query == "list repos with multiple users":
    data = audit_all_permissions()
    filtered = {k: v for k, v in data.items() if v['total_users'] > 1}
    return format_table(filtered)  # Chỉ trả repos có >1 user

if query == "find user X":
    data = find_user_across_repos(username=X)
    return format_list(data)  # Chỉ trả list, không table

if query == "add user":
    result = add_user_to_repo(...)
    return "✅ Success" if result['success'] else f"❌ {result['error']}"
```

### Best Practices

**✅ DO:**

- Gọi đúng tool cho đúng task
- Format data ngay sau khi nhận từ MCP
- Dùng tables cho multi-row data
- Dùng lists cho single-column data
- Trả lời ngắn gọn, chỉ thông tin được hỏi

**❌ DON'T:**

- Giải thích lại data (data tự giải thích)
- Thêm sections không cần thiết
- Tạo statistics nếu không được hỏi
- Repeat input của user
- Dùng markdown headers (##) trừ khi cần group data

### Error Handling

Khi gặp lỗi:

```
❌ [Tool name] failed: [error message]

Suggestions:
- Check if repo exists
- Verify token has required permissions
- Ensure username is correct
```

Không giải thích dài dòng, chỉ cung cấp actionable suggestions.

### Examples

#### ✅ GOOD Response:

```
User: "List repos và users"

Agent:
42 repos, 15 có collaborators:

| Repo | Users |
|------|-------|
| foxserfaty/StriderProject | foxserfaty(admin), LuuHung247(write), +3 more |
| Hancovirus/SpaceShooter | Hancovirus(admin), LuuHung247(write), +3 more |
... và 13 repos nữa
```

#### ❌ BAD Response:

```
User: "List repos và users"

Agent:
Báo cáo đầy đủ: Repositories và Users
Tổng số: 42 repositories

Repositories có nhiều collaborators
1. foxserfaty/StriderProject (Public) - 5 users
   foxserfaty (admin) ⭐
   LuuHung247 (write) ✏️
   tuananhlehuy1111 (write)
   ArsiHien (write)
   Hancovirus (write)

2. Hancovirus/SpaceShooter (Public) - 5 users
   Hancovirus (admin) ⭐
   ...

[100 more lines...]

Thống kê Users
Top collaborators xuất hiện nhiều nhất:
...

Tổng kết:
Tổng repositories: 42
Repos có collaborators: 15
...
```

### Token Usage Metrics

**Target benchmarks:**

| Query Type        | Expected Tokens | Max Tokens |
| ----------------- | --------------- | ---------- |
| List repos        | 200-500         | 1000       |
| Audit permissions | 500-1000        | 2000       |
| Add/Remove user   | 50-100          | 200        |
| Find user         | 100-300         | 500        |
| Bulk operations   | 300-800         | 1500       |

**Nếu response vượt Max Tokens → Cần refactor response format**

### Monitoring Commands

Để kiểm tra token usage:

```bash
# (Nội bộ - không output ra user)
[Check response length before sending]
[If >2000 tokens → compress data]
[If >5000 tokens → show top 10 only]
```

### Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│ QUICK REFERENCE - Agent Response Format             │
├─────────────────────────────────────────────────────┤
│ List/Audit  → Table                                 │
│ Add/Remove  → ✅/❌ + one line                       │
│ Find user   → List with roles                       │
│ Bulk ops    → Summary (X succeeded, Y failed)       │
│                                                     │
│ Rules:                                              │
│ - No narrative                                      │
│ - No statistics unless asked                        │
│ - Top 10 if >10 items                               │
│ - Tables for structured data                        │
└─────────────────────────────────────────────────────┘
```

---

## 🔍 Debugging

Nếu tool không hoạt động:

1. Check token permissions:

```bash
curl -H "Authorization: token $MY_GITHUB_TOKEN" https://api.github.com/user
```

2. Test connectivity:

```bash
uv run mcp_admin.py --test
```

3. Check logs:

- MCP server logs trong terminal
- GitHub API errors trong response

4. Verify environment:

```bash
echo $MY_GITHUB_TOKEN  # Should not be empty
```

## 📚 Additional Resources

- [GitHub REST API Docs](https://docs.github.com/en/rest)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [MCP Protocol Spec](https://modelcontextprotocol.io)
