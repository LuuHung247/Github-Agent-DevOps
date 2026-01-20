---
name: repo-from-template
description: "Tạo repository mới từ template có sẵn. Dùng khi: (1) User muốn tạo project mới từ template, (2) Clone structure của repo template, (3) Bootstrap dự án nhanh từ boilerplate."
---

# Create Repository from Template

Tạo repo mới từ template via MCP github-admin.

**LUÔN trả lời bằng tiếng Việt. PHẢI tuân thủ TỪNG BƯỚC trong workflow - KHÔNG được bỏ qua bước nào.**

## Quick Start

**Bước đầu tiên BẮT BUỘC:**

1. Chạy `quick_overview` → Lấy thông tin account
2. Chạy `list_template_repos()` → Hiển thị danh sách templates có sẵn trên account của user
3. Hỏi user chọn template hoặc nhập template từ owner khác

## Workflow

**PHẢI thực hiện đủ 4 bước theo thứ tự:**

1. **Template Discovery** → `list_template_repos()` hoặc `get_user_info(username=template_owner)`
2. **User Preferences** → Hỏi user về repo settings (nếu chưa chỉ định)
3. **Create Repo** → `create_repo_from_template(template_owner, template_repo, name, ...)`
4. **Verification** → `list_repos()` để confirm repo đã tạo thành công

### Bước 2: User Preferences (BẮT BUỘC)

**Nếu user CHƯA chỉ định rõ trong request → PHẢI dùng AskUserQuestion:**

```
Question 1: "Bạn muốn tạo repo public hay private?"
  Options:
    - Private → private=true
    - Public → private=false

Question 2: "Bạn có muốn copy tất cả branches từ template không?"
  Options:
    - Chỉ branch mặc định  → include_all_branches=false
    - Tất cả branches → include_all_branches=true
```

**Nếu user ĐÃ chỉ định (VD: "tạo public repo" hay "copy all branches"):**

- KHÔNG cần hỏi lại
- Sử dụng trực tiếp giá trị user đã chỉ định

## Template Selection Flow

**Nếu user KHÔNG chỉ định template cụ thể:**

- Chạy `list_template_repos()` để show templates trên account user
- Nếu có templates → Hiển thị danh sách cho user chọn
- Nếu không có templates → Hỏi user về template từ owner khác

**Nếu user chỉ định template (VD: "từ template X/Y"):**

- Skip `list_template_repos()`
- Verify trực tiếp qua `get_user_info(username=template_owner)`

## Parameters

| Param                  | Required | Mặc định | Mô tả                             |
| ---------------------- | -------- | -------- | --------------------------------- |
| `template_owner`       | ✅       | -        | Owner của template (username/org) |
| `template_repo`        | ✅       | -        | Tên template repository           |
| `name`                 | ✅       | -        | Tên repo mới                      |
| `description`          | ❌       | -        | Mô tả repo mới                    |
| `private`              | ❌       | `true`   | Private repo?                     |
| `include_all_branches` | ❌       | `false`  | Copy tất cả branches?             |

## Key Patterns

**Template sources:**

- **Own templates**: Dùng `list_template_repos()` để xem templates trên account user
- **External templates**: Verify owner qua `get_user_info(username)` trước

**Template requirements:**

- Template repo PHẢI được đánh dấu là "Template repository" trên GitHub
- Nếu template không public → user cần quyền truy cập

**Naming conventions & Auto-correction:**

- **Standard:** Ưu tiên `kebab-case` (chữ thường, nối bằng dấu gạch ngang)
- **Rules:**
  1. Chuyển tất cả về chữ thường (lowercase)
  2. Thay thế khoảng trắng bằng dấu gạch ngang (-)
  3. Loại bỏ dấu tiếng Việt và ký tự đặc biệt (VD: "Học Tập" → "hoc-tap")
- **Agent Behavior:** Nếu input không chuẩn → Tự động format lại trước khi gọi function

## Error Handling

| Error                 | Cách xử lý                                                |
| --------------------- | --------------------------------------------------------- |
| `NOT_FOUND`           | Verify template_owner + template_repo qua `get_user_info` |
| `PERMISSION_DENIED`   | Kiểm tra template visibility + token scope `repo`         |
| `VALIDATION_ERROR`    | Tự động format lại tên (sanitize) và thực hiện lại        |
| `NAME_ALREADY_EXISTS` | Đề xuất tên khác cho user hoặc hỏi tên mới                |
