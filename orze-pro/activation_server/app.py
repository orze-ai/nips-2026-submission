"""orze-pro activation server.

FastAPI app for managing license activations with per-key machine limits.
Deploy behind a reverse proxy at orze.ai/activate.

Endpoints:
    POST /activate      — Activate a key on a machine
    POST /deactivate    — Free a machine slot
    POST /verify        — Check if an activation token is still valid
    GET  /admin/keys    — List all keys and activations (admin)
    POST /admin/revoke  — Revoke activations by key or machine (admin)
"""

import base64
import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from db import (
    init_db, get_activations, create_activation, remove_activation,
    get_activation_by_token, touch_verified,
    revoke_by_key, revoke_by_machine, list_all_keys,
)

# --- Configuration ---

_PUBLIC_KEY_B64 = "s74ibUHhEHy5Sgf6PDICO/P+pFP+dnV/Wuitz4vuzAc="
_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
_DEFAULT_MAX_MACHINES = 3

app = FastAPI(title="orze-pro Activation Server", version="1.0.0")
_conn = None


def _db():
    global _conn
    if _conn is None:
        _conn = init_db()
    return _conn


# --- Key verification (same as client) ---

def _get_public_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    pub_bytes = base64.b64decode(_PUBLIC_KEY_B64)
    return Ed25519PublicKey.from_public_bytes(pub_bytes)


def _verify_key(key: str) -> Optional[dict]:
    """Verify key signature. Returns payload dict or None."""
    if not key or not key.startswith("ORZE-PRO-"):
        return None
    try:
        token = key[len("ORZE-PRO-"):]
        parts = token.rsplit(".", 1)
        if len(parts) != 2:
            return None
        payload_b64, sig_b64 = parts
        payload_b64_padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        sig_b64_padded = sig_b64 + "=" * (-len(sig_b64) % 4)
        signature = base64.urlsafe_b64decode(sig_b64_padded)
        public_key = _get_public_key()
        public_key.verify(signature, payload_b64.encode())
        payload_json = base64.urlsafe_b64decode(payload_b64_padded)
        payload = json.loads(payload_json)
        # Check expiry
        expires = payload.get("expires")
        if expires:
            try:
                if datetime.now() > datetime.strptime(expires, "%Y-%m-%d"):
                    return None
            except ValueError:
                pass
        return payload
    except Exception:
        return None


def _key_hash(key: str) -> str:
    """SHA256 hash of the full key for storage (never store key itself)."""
    return hashlib.sha256(key.encode()).hexdigest()


# --- PyPI auth (validates license key via Basic Auth) ---

@app.get("/pypi-auth")
def pypi_auth(authorization: str = Header(default="")):
    """Nginx auth_request subrequest — returns 200 if the license key is valid.

    pip sends Basic Auth with username ``__token__`` and password = license key.
    """
    if not authorization.lower().startswith("basic "):
        raise HTTPException(status_code=401, detail="Missing credentials")
    try:
        decoded = base64.b64decode(authorization.split(" ", 1)[1]).decode()
        _user, _sep, password = decoded.partition(":")
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed credentials")

    from fastapi.responses import Response

    # __admin__:<ADMIN_TOKEN> — for twine uploads
    if _user == "__admin__":
        if not _ADMIN_TOKEN or password != _ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid admin token")
        return Response(status_code=200)

    # __token__:<license key> — for pip downloads
    payload = _verify_key(password)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid license key")
    return Response(status_code=200)


# --- Request/response models ---

class ActivateRequest(BaseModel):
    key: str
    machine_id: str
    machine_name: Optional[str] = None


class DeactivateRequest(BaseModel):
    key: str
    machine_id: str


class VerifyRequest(BaseModel):
    token: str


class RevokeRequest(BaseModel):
    key: Optional[str] = None
    machine_id: Optional[str] = None


# --- Endpoints ---

@app.post("/activate")
def activate(req: ActivateRequest):
    """Activate a key on a machine."""
    payload = _verify_key(req.key)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid license key")

    kh = _key_hash(req.key)
    max_machines = payload.get("max_machines", _DEFAULT_MAX_MACHINES)
    current = get_activations(_db(), kh)

    # Check if this machine is already activated
    for act in current:
        if act["machine_id"] == req.machine_id:
            # Re-activate — update verification timestamp
            touch_verified(_db(), act["token"])
            return {
                "token": act["token"],
                "customer": payload.get("customer"),
                "tier": payload.get("tier"),
                "expires": payload.get("expires"),
                "machines_used": len(current),
                "machines_max": max_machines,
            }

    # Check machine limit
    if len(current) >= max_machines:
        machine_list = [
            {"machine_id": a["machine_id"], "machine_name": a["machine_name"],
             "activated_at": a["activated_at"]}
            for a in current
        ]
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Activation limit reached ({len(current)}/{max_machines}). "
                         "Deactivate another machine first.",
                "machines": machine_list,
            },
        )

    # Create new activation
    token = str(uuid.uuid4())
    create_activation(
        _db(), kh, payload.get("customer", ""),
        payload.get("tier", "pro"), req.machine_id,
        req.machine_name or "", token,
    )

    return {
        "token": token,
        "customer": payload.get("customer"),
        "tier": payload.get("tier"),
        "expires": payload.get("expires"),
        "machines_used": len(current) + 1,
        "machines_max": max_machines,
    }


@app.post("/deactivate")
def deactivate(req: DeactivateRequest):
    """Deactivate a key on a machine, freeing the slot."""
    payload = _verify_key(req.key)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid license key")

    kh = _key_hash(req.key)
    removed = remove_activation(_db(), kh, req.machine_id)
    if not removed:
        raise HTTPException(status_code=404, detail="No activation found for this machine")

    max_machines = payload.get("max_machines", _DEFAULT_MAX_MACHINES)
    remaining = get_activations(_db(), kh)
    return {
        "ok": True,
        "machines_used": len(remaining),
        "machines_max": max_machines,
    }


@app.post("/verify")
def verify(req: VerifyRequest):
    """Verify an activation token is still valid."""
    act = get_activation_by_token(_db(), req.token)
    if act is None:
        return {"valid": False}

    touch_verified(_db(), req.token)
    return {
        "valid": True,
        "customer": act["customer"],
        "tier": act["tier"],
    }


# --- Admin endpoints ---

def _require_admin(authorization: str = Header(default="")):
    if not _ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not configured")
    expected = f"Bearer {_ADMIN_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")


@app.get("/admin/keys")
def admin_keys(authorization: str = Header(default="")):
    """List all keys with their activations."""
    _require_admin(authorization)
    keys = list_all_keys(_db())
    # Add max_machines info (from key payload — we don't have it stored,
    # so we show what's in the DB)
    for k in keys:
        k["key_prefix"] = k["key_hash"][:16] + "..."
    return keys


@app.post("/admin/revoke")
def admin_revoke(req: RevokeRequest, authorization: str = Header(default="")):
    """Revoke activations by key or machine ID."""
    _require_admin(authorization)
    if not req.key and not req.machine_id:
        raise HTTPException(status_code=400, detail="Provide key or machine_id")

    revoked = 0
    if req.key:
        kh = _key_hash(req.key)
        revoked += revoke_by_key(_db(), kh)
    if req.machine_id:
        revoked += revoke_by_machine(_db(), req.machine_id)

    return {"ok": True, "revoked": revoked}
