---
name: repo-from-template
description: "Tạo repository mới từ template có sẵn. Dùng khi: (1) User muốn tạo project mới từ template, (2) Clone structure của repo template, (3) Bootstrap dự án nhanh từ boilerplate."
---

# Create Repository from Template

Tạo repo mới từ template via MCP github-admin.

**LUÔN trả lời bằng tiếng Việt.**

## MCP Tool

```
mcp__github-admin__create_repo_from_template
```

## Parameters

| Param | Required | Description |
|-------|----------|-------------|
| `template_owner` | ✅ | Owner của template (username/org) |
| `template_repo` | ✅ | Tên template repository |
| `name` | ✅ | Tên repo mới |
| `description` | ❌ | Mô tả repo mới |
| `private` | ❌ | Private? (default: `true`) |
| `include_all_branches` | ❌ | Copy tất cả branches? (default: `false`) |

## Workflow

**PHẢI thực hiện đủ 3 bước theo thứ tự:**

1. **Xác nhận template tồn tại** → `get_user_info(username=template_owner)` để verify owner
2. **Tạo repo** → `create_repo_from_template(template_owner, template_repo, name, ...)`
3. **Verify** → `list_repos()` để confirm repo đã được tạo

## Ví dụ

```
# Tạo private repo từ template
create_repo_from_template(
    template_owner="github",
    template_repo="gitignore",
    name="my-new-project",
    description="My awesome project",
    private=true
)

# Tạo public repo với tất cả branches
create_repo_from_template(
    template_owner="nextjs",
    template_repo="next.js",
    name="my-nextjs-app",
    private=false,
    include_all_branches=true
)
```

## Lưu Ý

- Template repo **PHẢI** được đánh dấu là "Template repository" trên GitHub
- Token cần scope `repo` để tạo private repos
- Nếu template không public, user cần quyền truy cập vào template

## Error Handling

| Error | Nguyên nhân | Cách xử lý |
|-------|-------------|------------|
| `NOT_FOUND` | Template không tồn tại | Verify template_owner + template_repo |
| `PERMISSION_DENIED` | Không có quyền access template | Kiểm tra template visibility |
| `VALIDATION_ERROR` | Repo name không hợp lệ | Chỉ dùng a-z, 0-9, -, _ |
