---
name: github-actions-secrets
description: "Quản lý GitHub Actions Environments & Environment Secrets. Dùng khi: (1) Tạo/xóa environment (production/staging/dev), (2) Tạo/xóa environment secrets, (3) Liệt kê environments và secrets, (4) Audit environment secrets."
---

# GitHub Actions Environment & Secrets Management

Quản lý GitHub Actions **Environments** và **Environment Secrets** qua MCP github-admin.

**LUÔN trả lời bằng tiếng Việt. PHẢI tuân thủ TỪNG BƯỚC trong workflow - KHÔNG được bỏ qua bước nào.**

## Quick Start

**Bước đầu tiên BẮT BUỘC:**
Chạy `quick_overview` để lấy thông tin account.

## Workflows

### 1. Tạo Environment

**PHẢI thực hiện đủ 2 bước theo thứ tự:**

1. `list_environments(repo="X")` → Check environment đã tồn tại chưa
2. `create_environment(repo, environment)` → Tạo environment

### 2. Tạo Environment Secret

**PHẢI thực hiện đủ 4 bước theo thứ tự:**

1. `list_environments(repo="X")` → Check environment tồn tại
2. Nếu chưa có → `create_environment(repo, environment)` để tạo trước
3. `list_environment_secrets(repo, environment)` → Check secret đã tồn tại chưa
4. Nếu secret đã tồn tại → HỎI user có muốn update không
5. `create_environment_secret(repo, environment, secret_name, secret_value)` → Tạo secret

**LƯU Ý:** Secret value được encrypt tự động, KHÔNG hiển thị trong response.

### 3. Xóa Environment Secret

**PHẢI thực hiện đủ 2 bước theo thứ tự:**

1. `list_environment_secrets(repo, environment)` → Liệt kê secrets
2. Confirm với user TRƯỚC khi xóa → `delete_environment_secret(repo, environment, secret_name)`

### 4. Xóa Environment

**PHẢI thực hiện đủ 3 bước theo thứ tự:**

1. `list_environments(repo="X")` → Liệt kê environments
2. `list_environment_secrets(repo, environment)` → Check có secrets trong environment không
3. Confirm với user TRƯỚC khi xóa → `delete_environment(repo, environment)`

**CẢNH BÁO:** Xóa environment sẽ xóa TẤT CẢ secrets trong environment đó!

### 5. Audit Environment Secrets

**PHẢI thực hiện đủ 2 bước:**

1. `list_environments(repo="X")` → Liệt kê TẤT CẢ environments
2. `list_environment_secrets(repo, environment)` → Cho MỖI environment

## MCP Tools

| Tool                          | Protected | Mô tả                                    |
| ----------------------------- | --------- | ---------------------------------------- |
| `list_environments`           | ❌         | Liệt kê environments                     |
| `create_environment`          | ✅         | Tạo/update environment                   |
| `delete_environment`          | ✅         | Xóa environment                          |
| `list_environment_secrets`    | ❌         | Liệt kê environment secrets              |
| `create_environment_secret`   | ✅         | Tạo/update environment secret            |
| `delete_environment_secret`   | ✅         | Xóa environment secret                   |

## Key Concepts

**Environment Secret là gì?**

- Secret gắn với 1 environment cụ thể (production, staging, dev)
- Chỉ workflows deploy đến environment đó mới truy cập được
- Ít rủi ro hơn vì secret bị cô lập theo môi trường

**Environment Settings:**

- **wait_timer**: Độ trễ trước khi deployment chạy (để rollback nếu cần)
- **reviewers**: Danh sách users cần approve trước khi deploy
- **deployment_branch_policy**: Giới hạn branch nào được deploy

**Secret naming rules:**
- Phải bắt đầu bằng chữ cái hoặc underscore
- Chỉ chứa: letters, digits, underscores, hyphens
- Agent tự động validate và báo lỗi nếu không hợp lệ

## Error Handling

| Error               | Cách xử lý                              |
| ------------------- | --------------------------------------- |
| `VALIDATION_ERROR`  | Kiểm tra secret/environment name format |
| `PERMISSION_DENIED` | Verify admin rights + repo access       |
| `NOT_FOUND`         | Environment hoặc repo không tồn tại     |
| `API_ERROR`         | Check PyNaCl installed (`pip install pynacl`) |
