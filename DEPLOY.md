# Deployment Security Checklist

## ✅ Security Measures

### Secrets Management
- ✅ **No secrets in code**: All sensitive values loaded from environment variables
- ✅ **No secrets in Docker**: `.env` and `.envrc` excluded via `.dockerignore`
- ✅ **No secrets in git**: `.env` excluded via `.gitignore`
- ✅ **No secrets in fly.toml**: Only public config (port, paths) in version control

### CORS Protection
- ✅ **Domain-restricted**: CORS configured via `CORS_ORIGINS` environment variable
- ✅ **Production default**: Only `butlerian.xyz` and `www.butlerian.xyz`
- ✅ **No wildcards**: Explicit domain whitelist, no `*` allowed
- ✅ **Limited methods**: Only GET, POST, OPTIONS (no PUT, DELETE, PATCH)
- ✅ **Limited headers**: Only Content-Type and Authorization

### Required Secrets (Set via Fly.io)
```bash
# OpenAI API key (REQUIRED)
fly secrets set OPENAI_API_KEY="sk-..."

# CORS origins (defaults to butlerian.xyz if not set)
fly secrets set CORS_ORIGINS="https://butlerian.xyz,https://www.butlerian.xyz"
```

### Environments (local vs cloud)

| Env / context | PHILOSOPH_ENV | CORS default |
|---------------|----------------|---------------|
| Local (e.g. `.env` with `PHILOSOPH_ENV=development`) | `development` | `localhost:8000`, `localhost:5713`, `127.0.0.1:*` |
| Fly.io (or `PHILOSOPH_ENV=production`) | `production` | `https://butlerian.xyz`, `https://www.butlerian.xyz` |

- **Local:** Set `PHILOSOPH_ENV=development` in `.env` (see `.env.example`). CORS then allows localhost by default; override with `CORS_ORIGINS` if needed.
- **Cloud:** Fly sets `FLY_APP_NAME`, so the app runs as production. `fly.toml` sets `PHILOSOPH_ENV=production` and `CORS_ORIGINS` explicitly.

### Local Development

**UI (Vite)** – serve the frontend on a local port with hot reload; API and WebSocket are proxied to the backend:
```bash
npm install && npm run dev
```
Opens http://localhost:5713 and proxies `/api`, `/health`, etc. to `http://localhost:8000`. Start the backend separately (e.g. `philosophai-server` or `uvicorn src.api.main:app --reload --port 8000`).

**CORS / env for local** – use development defaults (localhost allowed) by setting in `.env`:
```bash
PHILOSOPH_ENV=development
OPENAI_API_KEY=sk-...
```
To add more origins, set `CORS_ORIGINS` (comma-separated).

## Deployment Commands

### Initial Deploy
```bash
cd philosophai

# Create app
fly apps create philo-butlerian

# Create persistent volume for DuckDB
fly volumes create philo_data --size 3 --region iad

# Set secrets
fly secrets set OPENAI_API_KEY="sk-..."

# Deploy
fly deploy

# Add custom domain (apex often redirects www; philo subdomain may also point at the app — add to CORS if you use it)
fly certs create philo.butlerian.xyz
```

### Updates
```bash
fly deploy
```

### Verify Security
```bash
# Check environment variables (secrets hidden)
fly config show

# Test CORS (should reject unauthorized origins)
curl -H "Origin: https://malicious.com" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS https://philo.butlerian.xyz/api/query
```
