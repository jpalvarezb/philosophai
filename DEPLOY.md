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

### Local Development
For local development with CORS to localhost:
```bash
# .env (never commit this file)
OPENAI_API_KEY=sk-...
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,https://butlerian.xyz
```

## Deployment Commands

### Initial Deploy
```bash
cd philosiphai

# Create app
fly apps create philo-butlerian

# Create persistent volume for DuckDB
fly volumes create philo_data --size 3 --region iad

# Set secrets
fly secrets set OPENAI_API_KEY="sk-..."

# Deploy
fly deploy

# Add custom domain
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
