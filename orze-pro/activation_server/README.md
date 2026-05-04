# orze-pro Activation Server

Manages per-key machine activation limits for orze-pro licenses.

## Local development

```bash
pip install -r requirements.txt
ADMIN_TOKEN=secret uvicorn app:app --reload --port 8000
```

## Deploy

```bash
docker build -t orze-activation .
docker run -p 8000:8000 \
  -e ADMIN_TOKEN=your-secret-token \
  -v /data/activations:/app/activations.db \
  orze-activation
```

Put behind nginx/caddy at `orze.ai/activate` with TLS.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | /activate | license key | Activate a key on a machine |
| POST | /deactivate | license key | Free a machine slot |
| POST | /verify | token | Check activation token validity |
| GET | /admin/keys | admin bearer | List all keys and activations |
| POST | /admin/revoke | admin bearer | Revoke by key or machine |

## Environment variables

- `ADMIN_TOKEN` — Required for /admin/* endpoints
- Database is stored as `activations.db` in the working directory
