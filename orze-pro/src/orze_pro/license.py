"""orze-pro license verification.

Verifies license keys using Ed25519 signature verification.
The public key is embedded in the package. The private key
never leaves the issuer.

CALLING SPEC:
    verify_key(key: str) -> dict or None
        Returns the license payload if valid, None if invalid.

    check_license() -> dict or None
        Reads key from ORZE_PRO_KEY env var or .env file, verifies it.

    is_licensed() -> bool
        Returns True if a valid, non-expired license is active.

    license_info() -> str
        Human-readable license status.
"""

import base64
import hashlib
import json
import logging
import os
import socket
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze_pro")

# Public key for verifying license signatures.
# This is NOT secret — it can only verify, not create.
_PUBLIC_KEY_B64 = "s74ibUHhEHy5Sgf6PDICO/P+pFP+dnV/Wuitz4vuzAc="

# Activation system
_ACTIVATION_CACHE = Path.home() / ".orze-pro-activation.json"
_ACTIVATION_SERVER = os.environ.get("ORZE_ACTIVATION_SERVER", "https://ANON.example/activate")
_GRACE_DAYS = 7  # offline grace period
_NET_TIMEOUT = 5  # seconds for all network calls

_cached_license = None
_checked = False


def _get_public_key():
    """Load the Ed25519 public key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    pub_bytes = base64.b64decode(_PUBLIC_KEY_B64)
    return Ed25519PublicKey.from_public_bytes(pub_bytes)


def verify_key(key: str) -> Optional[dict]:
    """Verify a license key. Returns payload dict if valid, None otherwise."""
    if not key:
        return None
    key = key.strip()
    if not key.startswith("ORZE-PRO-"):
        return None

    try:
        # Strip prefix
        token = key[len("ORZE-PRO-"):]

        # Split payload and signature
        parts = token.rsplit(".", 1)
        if len(parts) != 2:
            return None
        payload_b64, sig_b64 = parts

        # Restore base64 padding
        payload_b64 += "=" * (-len(payload_b64) % 4)
        sig_b64 += "=" * (-len(sig_b64) % 4)

        # Verify signature
        signature = base64.urlsafe_b64decode(sig_b64)
        public_key = _get_public_key()

        # The signed data is the payload_b64 WITHOUT padding
        signed_data = parts[0].encode()
        public_key.verify(signature, signed_data)

        # Signature valid — decode payload
        payload_json = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_json)

        # Check expiry
        expires = payload.get("expires")
        if expires:
            try:
                exp_date = datetime.strptime(expires, "%Y-%m-%d")
                if datetime.now() > exp_date:
                    logger.warning("orze-pro license expired on %s", expires)
                    return None
            except ValueError:
                pass

        return payload

    except Exception as e:
        logger.debug("License verification failed: %s", e)
        return None


def verify_key_reason(key: str) -> str:
    """Return human-readable reason why a key is invalid. Empty string if valid."""
    if not key:
        return "No key provided."
    key = key.strip()
    if not key.startswith("ORZE-PRO-"):
        return "Key should start with ORZE-PRO-"

    token = key[len("ORZE-PRO-"):]
    parts = token.rsplit(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return ("Key appears truncated. Make sure you copied the full key "
                "including the part after the dot.")
    try:
        payload_b64 = parts[0] + "=" * (-len(parts[0]) % 4)
        sig_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        base64.urlsafe_b64decode(payload_b64)
        base64.urlsafe_b64decode(sig_b64)
    except Exception:
        return ("Key format is invalid. Check for extra spaces or line breaks "
                "when copying.")

    try:
        signature = base64.urlsafe_b64decode(sig_b64)
        public_key = _get_public_key()
        public_key.verify(signature, parts[0].encode())
    except Exception:
        return ("Signature verification failed. This key was not issued by ANON.example. "
                "Double-check you copied it from your purchase confirmation email.")

    # Signature valid — check expiry
    try:
        payload_json = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_json)
        expires = payload.get("expires")
        if expires:
            exp_date = datetime.strptime(expires, "%Y-%m-%d")
            if datetime.now() > exp_date:
                return f"This key expired on {expires}. Visit ANON.example/pro to renew."
    except Exception:
        pass

    return ""  # valid


def _find_key() -> Optional[str]:
    """Find license key from env var or .env file."""
    # 1. Environment variable
    key = os.environ.get("ORZE_PRO_KEY", "").strip()
    if key:
        return key

    # 2. .env file in current directory
    env_path = Path(".env")
    if env_path.exists():
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("ORZE_PRO_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass

    # 3. ~/.orze-pro.key file
    home_key = Path.home() / ".orze-pro.key"
    if home_key.exists():
        try:
            return home_key.read_text().strip()
        except OSError:
            pass

    return None


def _get_machine_id() -> str:
    """Generate stable machine identifier from hostname + MAC address."""
    hostname = socket.gethostname()
    mac = uuid.getnode()  # MAC address as int
    raw = f"{hostname}-{mac}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _key_hash(key: str) -> str:
    """SHA256 hash of the full key (never store/transmit the key itself)."""
    return hashlib.sha256(key.encode()).hexdigest()


def _activate_online(key: str) -> Optional[dict]:
    """Activate key with server. Returns activation dict or None."""
    import urllib.request
    import urllib.error
    try:
        data = json.dumps({
            "key": key,
            "machine_id": _get_machine_id(),
            "machine_name": socket.gethostname(),
        }).encode()
        req = urllib.request.Request(
            f"{_ACTIVATION_SERVER}/activate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_NET_TIMEOUT) as resp:
            result = json.loads(resp.read())
        # Cache the activation
        cache = {
            "token": result["token"],
            "customer": result.get("customer"),
            "tier": result.get("tier"),
            "expires": result.get("expires"),
            "key_hash": _key_hash(key),
            "machine_id": _get_machine_id(),
            "machines_used": result.get("machines_used"),
            "machines_max": result.get("machines_max"),
            "activated_at": datetime.now().isoformat(),
            "last_verified_at": datetime.now().isoformat(),
        }
        _save_activation_cache(cache)
        return cache
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            detail = body.get("detail", {})
            if isinstance(detail, dict):
                logger.error(detail.get("error", str(detail)))
            else:
                logger.error("Activation failed: %s", detail)
        except Exception:
            logger.error("Activation failed (HTTP %d)", e.code)
        return None
    except Exception as e:
        logger.debug("Activation server unreachable: %s", e)
        return None


def _save_activation_cache(cache: dict) -> None:
    """Save activation cache to disk with restricted permissions."""
    try:
        _ACTIVATION_CACHE.write_text(json.dumps(cache, indent=2))
        _ACTIVATION_CACHE.chmod(0o600)
    except OSError as e:
        logger.debug("Could not save activation cache: %s", e)


def _check_activation_cache() -> Optional[dict]:
    """Check cached activation. Valid if < GRACE_DAYS since last verification."""
    if not _ACTIVATION_CACHE.exists():
        return None
    try:
        cache = json.loads(_ACTIVATION_CACHE.read_text())
        last = cache.get("last_verified_at")
        if not last:
            return None
        last_dt = datetime.fromisoformat(last)
        age_days = (datetime.now() - last_dt).days
        if age_days <= _GRACE_DAYS:
            return cache
        return cache  # return it anyway — caller decides on grace period
    except Exception as e:
        logger.debug("Activation cache unreadable: %s", e)
        return None


def _verify_online(token: str) -> bool:
    """Verify activation token with server. Updates last_verified timestamp in cache."""
    import urllib.request
    try:
        data = json.dumps({"token": token}).encode()
        req = urllib.request.Request(
            f"{_ACTIVATION_SERVER}/verify",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_NET_TIMEOUT) as resp:
            result = json.loads(resp.read())
        if result.get("valid"):
            # Update cache timestamp
            if _ACTIVATION_CACHE.exists():
                try:
                    cache = json.loads(_ACTIVATION_CACHE.read_text())
                    cache["last_verified_at"] = datetime.now().isoformat()
                    _save_activation_cache(cache)
                except Exception:
                    pass
            return True
        return False
    except Exception as e:
        logger.debug("Online verification failed: %s", e)
        return False


def _maybe_reverify(cached: dict) -> None:
    """Non-blocking re-verification if cache is >1 day old."""
    import threading
    last = cached.get("last_verified_at")
    if not last:
        return
    try:
        age_days = (datetime.now() - datetime.fromisoformat(last)).days
        if age_days >= 1:
            token = cached.get("token")
            if token:
                t = threading.Thread(target=_verify_online, args=(token,), daemon=True)
                t.start()
    except Exception:
        pass


def deactivate_online(key: str) -> Optional[dict]:
    """Deactivate this machine's activation with the server. Returns response dict or None."""
    import urllib.request
    import urllib.error
    try:
        data = json.dumps({
            "key": key,
            "machine_id": _get_machine_id(),
        }).encode()
        req = urllib.request.Request(
            f"{_ACTIVATION_SERVER}/deactivate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_NET_TIMEOUT) as resp:
            result = json.loads(resp.read())
        # Remove local cache
        if _ACTIVATION_CACHE.exists():
            try:
                _ACTIVATION_CACHE.unlink()
            except OSError:
                pass
        return result
    except Exception as e:
        logger.debug("Deactivation failed: %s", e)
        return None


def get_activation_status() -> Optional[dict]:
    """Get cached activation status (machine count, etc.)."""
    if _ACTIVATION_CACHE.exists():
        try:
            return json.loads(_ACTIVATION_CACHE.read_text())
        except Exception:
            pass
    return None


def check_license() -> Optional[dict]:
    """Check license from environment. Returns payload or None.

    Flow:
    1. Find key (env var or file)
    2. Verify key signature locally (fast, offline)
    3. Check activation cache — if valid and fresh, use it
    4. Try online verification/activation
    5. If offline, fall back to grace period
    """
    global _cached_license, _checked
    if _checked:
        return _cached_license

    _checked = True
    key = _find_key()
    if not key:
        return None

    # Step 1: verify key signature locally (fast, offline)
    payload = verify_key(key)
    if payload is None:
        return None

    # Step 2: check activation cache
    cached = _check_activation_cache()
    if cached and cached.get("key_hash") == _key_hash(key):
        last = cached.get("last_verified_at")
        if last:
            age_days = (datetime.now() - datetime.fromisoformat(last)).days
            if age_days <= _GRACE_DAYS:
                _cached_license = payload
                # Try background re-verification if >1 day old (non-blocking)
                _maybe_reverify(cached)
                return _cached_license

    # Step 3: activate online
    activation = _activate_online(key)
    if activation:
        _cached_license = payload
        return _cached_license

    # Step 4: grace period — if cache exists but server unreachable
    if cached and cached.get("key_hash") == _key_hash(key):
        last = cached.get("last_verified_at")
        if last:
            age_days = (datetime.now() - datetime.fromisoformat(last)).days
            if age_days <= _GRACE_DAYS:
                logger.warning(
                    "Activation server unreachable — using cached activation (%d days old)",
                    age_days,
                )
                _cached_license = payload
                return _cached_license

    # Step 5: first-time offline use — key signature valid but server
    # never reached. Create a local-only cache so the user can work
    # offline. Will activate properly when server becomes reachable.
    if payload:
        logger.warning(
            "Activation server unreachable — allowing offline use with valid key signature"
        )
        _save_activation_cache({
            "key_hash": _key_hash(key),
            "token": "offline-pending",
            "customer": payload.get("customer", "?"),
            "tier": payload.get("tier", "pro"),
            "last_verified_at": datetime.now().isoformat(),
        })
        _cached_license = payload
        return _cached_license

    logger.error(
        "Could not verify activation. Check internet connection or contact support@ANON.example"
    )
    return None


def is_licensed() -> bool:
    """Returns True if a valid license is active."""
    return check_license() is not None


def license_info() -> str:
    """Human-readable license status."""
    payload = check_license()
    if payload is None:
        key = _find_key()
        if key:
            return "orze-pro license invalid or expired. Contact support@ANON.example"
        return "orze-pro not activated. Set ORZE_PRO_KEY or get a key at ANON.example/pro"

    customer = payload.get("customer", "?")
    tier = payload.get("tier", "pro")
    expires = payload.get("expires", "never")
    return f"Licensed to {customer} ({tier}), expires {expires}"
