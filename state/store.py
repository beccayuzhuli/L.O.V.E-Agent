"""
================================================================================
L.O.V.E. Relationship Support Agent — SQLite Store
RSM 8430 Group 18
================================================================================

Provides all database read/write helpers used by the agent and app layers.
Call init_db() once at app startup; all other functions are safe to call anytime.

Usage:
    from state.store import init_db, create_session, append_message, ...
    init_db()
    sid = create_session()
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = Path("data/love_agent.db")
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _now() -> str:
    """UTC timestamp string for SQLite columns."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Return a connection with row-factory enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables from schema.sql if they don't already exist."""
    schema_sql = SCHEMA_PATH.read_text()
    conn = _get_conn()
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session() -> str:
    """Create a new session and return its ID."""
    session_id = uuid.uuid4().hex[:12]
    now = _now()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return session_id


def get_or_create_session(session_id: str | None = None) -> str:
    """
    If session_id is provided and exists, return it.
    Otherwise create a new session.
    """
    if session_id:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        if row:
            return str(row["session_id"])
    return create_session()


def list_sessions() -> list[dict[str, Any]]:
    """Return all sessions ordered by most recent first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def touch_session(session_id: str) -> None:
    """Update the updated_at timestamp for a session."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (_now(), session_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def append_message(session_id: str, role: str, content: str,
                   intent: str | None = None) -> None:
    """Append a message to the conversation history."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, intent, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, intent, _now()),
        )
        conn.commit()
        # Also bump the session timestamp.
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (_now(), session_id),
        )
        conn.commit()
    finally:
        conn.close()


def load_messages(session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Load the most recent `limit` messages for a session, oldest first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT role, content, intent, created_at FROM ("
            "  SELECT role, content, intent, created_at, id FROM messages "
            "  WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT ?"
            ") ORDER BY created_at ASC, id ASC",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Action State (multi-turn slot filling)
# ---------------------------------------------------------------------------

def save_action_state(session_id: str, current_intent: str,
                      slots: dict[str, Any]) -> None:
    """Upsert the in-progress action state for a session."""
    slots_json = json.dumps(slots, ensure_ascii=False)
    now = _now()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO action_state (session_id, current_intent, slots_json, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "  current_intent = excluded.current_intent, "
            "  slots_json = excluded.slots_json, "
            "  updated_at = excluded.updated_at",
            (session_id, current_intent, slots_json, now),
        )
        conn.commit()
    finally:
        conn.close()


def load_action_state(session_id: str) -> dict[str, Any] | None:
    """Load current action state, or None if no action is in progress."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT current_intent, slots_json, updated_at FROM action_state "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "current_intent": row["current_intent"],
            "slots": json.loads(row["slots_json"]),
            "updated_at": row["updated_at"],
        }
    finally:
        conn.close()


def clear_action_state(session_id: str) -> None:
    """Remove action state once a multi-turn action completes or is cancelled."""
    conn = _get_conn()
    try:
        conn.execute(
            "DELETE FROM action_state WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Saved Plans
# ---------------------------------------------------------------------------

def save_plan(session_id: str, plan: dict[str, Any],
              label: str = "default") -> int:
    """Persist a generated plan. Returns the row id."""
    plan_json = json.dumps(plan, ensure_ascii=False)
    now = _now()
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO saved_plans (session_id, label, plan_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, label, plan_json, now, now),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def get_saved_plans(session_id: str, label: str | None = None) -> list[dict[str, Any]]:
    """Retrieve saved plans for a session, optionally filtered by label."""
    conn = _get_conn()
    try:
        if label:
            rows = conn.execute(
                "SELECT id, label, plan_json, created_at FROM saved_plans "
                "WHERE session_id = ? AND label = ? ORDER BY created_at DESC",
                (session_id, label),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, label, plan_json, created_at FROM saved_plans "
                "WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "label": r["label"],
                "plan": json.loads(r["plan_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_latest_saved_plan(session_id: str) -> dict[str, Any] | None:
    """Return the most recently saved plan, or None."""
    plans = get_saved_plans(session_id)
    return plans[0] if plans else None


# ---------------------------------------------------------------------------
# User Profile Memory
# ---------------------------------------------------------------------------

def load_user_profile(session_id: str) -> dict[str, Any]:
    """Load structured profile memory for a session (empty dict if none)."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT profile_json FROM user_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return {}
        return json.loads(row["profile_json"])
    finally:
        conn.close()


def save_user_profile(session_id: str, profile: dict[str, Any]) -> None:
    """Upsert structured profile memory for a session."""
    profile_json = json.dumps(profile, ensure_ascii=False)
    now = _now()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO user_profiles (session_id, profile_json, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "  profile_json = excluded.profile_json, "
            "  updated_at = excluded.updated_at",
            (session_id, profile_json, now),
        )
        conn.commit()
    finally:
        conn.close()
