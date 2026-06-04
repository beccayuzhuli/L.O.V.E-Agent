"""
================================================================================
L.O.V.E. Relationship Support Agent — Session Memory Helpers
RSM 8430 Group 18
================================================================================

Lightweight helpers that manage in-session conversation memory and action state.
These wrap state/store.py functions and add convenience methods used by the
agent layer (router, actions) and the app layer (main.py).

Usage:
    from agent.memory import SessionMemory
    mem = SessionMemory(session_id)
    mem.add_user_message("I keep fighting with my partner")
    history = mem.get_history()
"""

from __future__ import annotations

import re
from typing import Any

from state.store import (
    append_message,
    clear_action_state,
    load_action_state,
    load_messages,
    load_user_profile,
    save_action_state,
    save_user_profile,
)

_PARTNER_LABELS: dict[str, str] = {
    "husband": "husband",
    "wife": "wife",
    "boyfriend": "boyfriend",
    "girlfriend": "girlfriend",
    "partner": "partner",
    "ex": "ex-partner",
}

_FOCUS_PATTERNS: list[tuple[str, str]] = [
    (r"\btrust|cheat|betray|lied\b", "trust"),
    (r"\bargu|fight|conflict\b", "conflict"),
    (r"\bcommunicat|listen|heard|misunderstood\b", "communication"),
    (r"\bboundar|respect\b", "boundaries"),
    (r"\bintimacy|sex|affection|physical\b", "intimacy"),
    (r"\bbreak\s?up|divorce|separat|ex\b", "breakup"),
    (r"\bneglect|distance|disconnected|growing apart\b", "emotional_distance"),
]

_EMOTION_PATTERNS: list[tuple[str, str]] = [
    (r"\bfrustrat|annoyed|angry\b", "frustrated"),
    (r"\bhurt|sad|heartbroken\b", "hurt"),
    (r"\banxious|worried|nervous\b", "anxious"),
    (r"\boverwhelm|drained|exhausted\b", "overwhelmed"),
    (r"\bguilty|regret\b", "guilty"),
]


def _dedupe_keep_order(items: list[str], max_items: int = 6) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item:
            continue
        key = item.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item.strip())
        if len(ordered) >= max_items:
            break
    return ordered


def _infer_profile_updates(text: str) -> dict[str, Any]:
    """Extract lightweight profile signals from a user message."""
    updates: dict[str, Any] = {}
    lower = text.lower()

    for token, label in _PARTNER_LABELS.items():
        if re.search(rf"\b{re.escape(token)}\b", lower):
            updates["relationship_label"] = label
            break

    focus_areas = [label for pattern, label in _FOCUS_PATTERNS if re.search(pattern, lower)]
    if focus_areas:
        updates["focus_areas"] = focus_areas

    emotions = [label for pattern, label in _EMOTION_PATTERNS if re.search(pattern, lower)]
    if emotions:
        updates["recent_emotions"] = emotions

    tone_match = re.search(
        r"\b(calm|gentle|direct|assertive|soft|collaborative|empathetic|firm)\b",
        lower,
    )
    if tone_match and ("tone" in lower or "come across" in lower or "sound" in lower):
        updates["preferred_tone"] = tone_match.group(1)

    goal_match = re.search(r"\bi want ([^.?!]{8,160})", text, re.IGNORECASE)
    if goal_match:
        updates["stated_goal"] = goal_match.group(1).strip()

    return updates


class SessionMemory:
    """Manages conversation history and action state for a single session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def add_user_message(self, content: str, intent: str | None = None) -> None:
        append_message(self.session_id, "user", content, intent=intent)

    def add_assistant_message(self, content: str, intent: str | None = None) -> None:
        append_message(self.session_id, "assistant", content, intent=intent)

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return conversation history as a list of {role, content} dicts."""
        return load_messages(self.session_id, limit=limit)

    def get_history_for_prompt(self, limit: int = 10) -> str:
        """
        Return recent messages formatted as a string for LLM context.
        Keeps token usage manageable by limiting to the last `limit` messages.
        """
        messages = self.get_history(limit=limit)
        lines: list[str] = []
        for m in messages:
            role = m["role"].capitalize()
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Multi-turn action state (slot filling)
    # ------------------------------------------------------------------

    def get_action_state(self) -> dict[str, Any] | None:
        """Get the current in-progress action, or None."""
        return load_action_state(self.session_id)

    def set_action_state(self, intent: str, slots: dict[str, Any]) -> None:
        """Save or update the in-progress action state."""
        save_action_state(self.session_id, intent, slots)

    def clear_action_state(self) -> None:
        """Clear action state once the action completes."""
        clear_action_state(self.session_id)

    # ------------------------------------------------------------------
    # Structured user profile memory (long-term within session)
    # ------------------------------------------------------------------

    def get_user_profile(self) -> dict[str, Any]:
        return load_user_profile(self.session_id)

    def update_user_profile(self, updates: dict[str, Any]) -> dict[str, Any]:
        current = self.get_user_profile()
        merged = dict(current)

        scalar_keys = ["relationship_label", "preferred_tone", "stated_goal"]
        for key in scalar_keys:
            value = updates.get(key)
            if value:
                merged[key] = value

        for key in ["focus_areas", "recent_emotions"]:
            existing = list(merged.get(key, []))
            incoming = list(updates.get(key, []))
            if existing or incoming:
                merged[key] = _dedupe_keep_order(existing + incoming)

        save_user_profile(self.session_id, merged)
        return merged

    def infer_and_update_profile(self, user_input: str) -> dict[str, Any]:
        updates = _infer_profile_updates(user_input)
        if not updates:
            return self.get_user_profile()
        return self.update_user_profile(updates)

    def get_profile_for_prompt(self) -> str:
        profile = self.get_user_profile()
        if not profile:
            return "No durable profile signals captured yet."

        lines: list[str] = []
        if profile.get("relationship_label"):
            lines.append(f"Relationship context: {profile['relationship_label']}")
        if profile.get("focus_areas"):
            lines.append(f"Recurring focus areas: {', '.join(profile['focus_areas'])}")
        if profile.get("recent_emotions"):
            lines.append(f"Recent emotions expressed: {', '.join(profile['recent_emotions'])}")
        if profile.get("preferred_tone"):
            lines.append(f"Preferred tone: {profile['preferred_tone']}")
        if profile.get("stated_goal"):
            lines.append(f"Stated goal: {profile['stated_goal']}")

        return "\n".join(lines) if lines else "No durable profile signals captured yet."

    # ------------------------------------------------------------------
    # Convenience: latest generated plan (in Streamlit session_state)
    # ------------------------------------------------------------------
    # The "latest plan" is stored in st.session_state by actions.py
    # and persisted to SQLite via state/store.save_plan().
    # These helpers exist so the agent layer can check without
    # importing Streamlit directly.

    @staticmethod
    def extract_latest_plan_from_state(st_session_state: dict) -> dict[str, Any] | None:
        """Pull the latest generated plan from Streamlit session state."""
        return st_session_state.get("latest_plan")
