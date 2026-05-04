"""SQLite database for orze-pro activation tracking.

Calling spec:
    init_db() -> sqlite3.Connection
    get_activations(conn, key_hash) -> list[dict]
    create_activation(conn, key_hash, customer, tier, machine_id, machine_name, token) -> dict
    remove_activation(conn, key_hash, machine_id) -> bool
    get_activation_by_token(conn, token) -> dict | None
    revoke_by_key(conn, key_hash) -> int
    revoke_by_machine(conn, machine_id) -> int
    list_all_keys(conn) -> list[dict]
    touch_verified(conn, token) -> None
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "activations.db"


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Initialize the database and return a connection."""
    conn = sqlite3.connect(str(db_path or DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activations (
            id INTEGER PRIMARY KEY,
            key_hash TEXT NOT NULL,
            customer TEXT NOT NULL,
            tier TEXT NOT NULL,
            machine_id TEXT NOT NULL,
            machine_name TEXT,
            token TEXT NOT NULL UNIQUE,
            activated_at TEXT NOT NULL,
            last_verified_at TEXT,
            revoked INTEGER DEFAULT 0,
            UNIQUE(key_hash, machine_id)
        )
    """)
    conn.commit()
    return conn


def get_activations(conn: sqlite3.Connection, key_hash: str) -> list[dict]:
    """Get all active (non-revoked) activations for a key."""
    rows = conn.execute(
        "SELECT * FROM activations WHERE key_hash = ? AND revoked = 0",
        (key_hash,),
    ).fetchall()
    return [dict(r) for r in rows]


def create_activation(conn: sqlite3.Connection, key_hash: str, customer: str,
                      tier: str, machine_id: str, machine_name: str,
                      token: str) -> dict:
    """Create a new activation record. Returns the activation dict."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO activations "
        "(key_hash, customer, tier, machine_id, machine_name, token, activated_at, last_verified_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (key_hash, customer, tier, machine_id, machine_name, token, now, now),
    )
    conn.commit()
    return {
        "key_hash": key_hash, "customer": customer, "tier": tier,
        "machine_id": machine_id, "machine_name": machine_name,
        "token": token, "activated_at": now, "last_verified_at": now,
    }


def remove_activation(conn: sqlite3.Connection, key_hash: str, machine_id: str) -> bool:
    """Revoke activation for a specific machine. Returns True if found."""
    cur = conn.execute(
        "UPDATE activations SET revoked = 1 WHERE key_hash = ? AND machine_id = ? AND revoked = 0",
        (key_hash, machine_id),
    )
    conn.commit()
    return cur.rowcount > 0


def get_activation_by_token(conn: sqlite3.Connection, token: str) -> dict | None:
    """Look up an activation by its token."""
    row = conn.execute(
        "SELECT * FROM activations WHERE token = ? AND revoked = 0",
        (token,),
    ).fetchone()
    return dict(row) if row else None


def touch_verified(conn: sqlite3.Connection, token: str) -> None:
    """Update last_verified_at timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE activations SET last_verified_at = ? WHERE token = ?",
        (now, token),
    )
    conn.commit()


def revoke_by_key(conn: sqlite3.Connection, key_hash: str) -> int:
    """Revoke all activations for a key hash. Returns count revoked."""
    cur = conn.execute(
        "UPDATE activations SET revoked = 1 WHERE key_hash = ? AND revoked = 0",
        (key_hash,),
    )
    conn.commit()
    return cur.rowcount


def revoke_by_machine(conn: sqlite3.Connection, machine_id: str) -> int:
    """Revoke all activations for a machine ID. Returns count revoked."""
    cur = conn.execute(
        "UPDATE activations SET revoked = 1 WHERE machine_id = ? AND revoked = 0",
        (machine_id,),
    )
    conn.commit()
    return cur.rowcount


def list_all_keys(conn: sqlite3.Connection) -> list[dict]:
    """List all keys with their activations (for admin)."""
    rows = conn.execute(
        "SELECT DISTINCT key_hash, customer, tier FROM activations"
    ).fetchall()
    result = []
    for r in rows:
        activations = get_activations(conn, r["key_hash"])
        result.append({
            "key_hash": r["key_hash"],
            "customer": r["customer"],
            "tier": r["tier"],
            "activations": activations,
            "machines_used": len(activations),
        })
    return result
