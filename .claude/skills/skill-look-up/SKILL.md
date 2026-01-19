---
name: skill-lookup-skillsmp
description: Activates when the user asks about Agent Skills, wants to find reusable AI capabilities, needs to install skills, or mentions skills for Claude. Use for discovering, retrieving, and installing skills.
---

# Getting started

- The skill require API key for access to skillsmp. As the user provide the key, store it in `.claude/skillsmp_api_key`.

# API reference

## GET /api/v1/skills/search

Search skills using keywords

| Parameter | Type   | Required | Description                            |
| --------- | ------ | -------- | -------------------------------------- |
| q         | string | ✓        | Search query                           |
| page      | number | -        | Page number (default: 1)               |
| limit     | number | -        | Items per page (default: 20, max: 100) |
| sortBy    | string | -        | Sort: stars \| recent                  |

## GET /api/v1/skills/ai-search

AI semantic search powered by Cloudflare AI

| Parameter | Type   | Required | Description     |
| --------- | ------ | -------- | --------------- |
| q         | string | ✓        | AI search query |

# Example

## Keyword Search

```bash
curl -X GET "https://skillsmp.com/api/v1/skills/search?q=SEO" \
  -H "Authorization: Bearer <your api key>"
```

## AI Semantic Search

```bash
curl -X GET "https://skillsmp.com/api/v1/skills/ai-search?q=How+to+create+a+web+scraper" \
  -H "Authorization: Bearer <your api key>"
```

# Error Handling

The API uses standard HTTP status codes and returns error details in JSON format.

| Error Code      | HTTP | Description                      |
| --------------- | ---- | -------------------------------- |
| MISSING_API_KEY | 401  | API key not provided             |
| INVALID_API_KEY | 401  | Invalid API key                  |
| MISSING_QUERY   | 400  | Missing required query parameter |
| INTERNAL_ERROR  | 500  | Internal server error            |

```json
{
  "success": false,
  "error": {
    "code": "INVALID_API_KEY",
    "message": "The provided API key is invalid"
  }
}
```