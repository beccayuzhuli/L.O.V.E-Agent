"""
================================================================================
L.O.V.E. Relationship Support Agent — Actions
RSM 8430 Group 18
================================================================================

Implements the three agent actions:
  A. build_plan    — Multi-turn: collects issue, goal, tone → structured plan
  B. reflection    — Single-turn: generates guided reflection prompts
  C. save / retrieve plan — Single-turn: persists or fetches plans from SQLite

All actions return a dict with a "type" key indicating the result kind, so the
UI layer can render them appropriately.

Usage:
    from agent.actions import handle_build_plan, handle_reflection,
                              handle_save_plan, handle_retrieve_plan
"""

from __future__ import annotations

import re
from typing import Any, Callable

from agent.memory import SessionMemory
from app.prompts import BUILD_PLAN_FROM_CONTEXT_PROMPT, BUILD_PLAN_PROMPT, REFLECTION_PROMPT
from state.store import (
    get_latest_saved_plan,
    get_saved_plans,
    save_plan as db_save_plan,
)


# ============================================================================
# A. Build Conversation Plan (MULTI-TURN)
# ============================================================================

# Required and optional slots for the conversation plan.
REQUIRED_SLOTS = ["issue", "goal", "tone"]
OPTIONAL_SLOTS = ["delivery_mode"]

_SLOT_QUESTIONS: dict[str, str] = {
    "issue": (
        "I'd love to help you prepare for that. "
        "First — what's the main thing you want to talk to them about? "
        "Just describe it in your own words, like *\"We keep arguing about "
        "household responsibilities\"* or *\"I feel like we're growing apart.\"*"
    ),
    "goal": (
        "Got it, thanks for sharing that. "
        "Now — what's the outcome you're hoping for? What would a good "
        "version of this conversation look like for you? "
        "For example: *\"I want us to agree on something fair\"* or "
        "*\"I just want them to understand how I feel.\"*"
    ),
    "tone": (
        "That makes sense. Last thing — what kind of tone do you want to set? "
        "Think about how you want to come across. Something like "
        "*calm and collaborative*, *gentle but direct*, or *assertive but caring*?"
    ),
}


# Minimum number of user messages before we consider the context "rich enough"
# to skip slot-filling and generate a plan directly from conversation history.
_CONTEXT_RICH_THRESHOLD = 3


def _has_rich_context(memory: SessionMemory) -> bool:
    """Check if the conversation has enough history to generate a plan directly."""
    history = memory.get_history(limit=50)
    user_messages = [m for m in history if m["role"] == "user"]
    return len(user_messages) >= _CONTEXT_RICH_THRESHOLD


def _looks_like_plan_request(text: str) -> bool:
    """Detect whether a message is primarily requesting plan generation."""
    lower = text.lower().strip()
    patterns = [
        r"\b(build|create|make)\s+(a\s+|my\s+)?(conversation\s+)?plan\b",
        r"\bhelp\s+me\s+(build|create|make|prepare|with)\s+(a\s+)?(conversation\s+)?plan\b",
        r"\bprepare\s+(for\s+)?(a\s+)?(difficult|hard|tough)?\s*conversation\b",
        r"\bconversation\s+plan\b",
    ]
    return any(re.search(pattern, lower) for pattern in patterns)


def _extract_issue_from_plan_request(text: str) -> str | None:
    """
    Extract an issue from natural messages like:
    "help me build a plan about feeling unheard in my relationship".
    """
    match = re.search(r"\b(?:about|regarding|on)\b[:\s]+(.+)", text, re.IGNORECASE)
    if not match:
        return None

    issue = match.group(1).strip().strip(".!?")
    if len(issue.split()) < 3:
        return None
    if _looks_like_plan_request(issue):
        return None
    return issue


def handle_build_plan(
    user_input: str,
    memory: SessionMemory,
    llm_fn: Callable[[str, str], str],
) -> dict[str, Any]:
    """
    Build a conversation plan. Two paths:

    1. CONTEXT-AWARE (rich history): If the user has been chatting for a while
       (3+ user messages), skip slot-filling and generate a plan directly from
       the full conversation history. This also handles "adjust it to my
       situation" requests.

    2. COLD-START (no history): Use multi-turn slot-filling to collect issue,
       goal, and tone before generating.

    Returns one of:
      {"type": "follow_up", "message": "...", "slots": {...}}
      {"type": "plan", "plan": {...}, "message": "..."}
    """
    state = memory.get_action_state()

    # --- Path 1: Context-aware plan from conversation history ---
    # Only use this path when we are NOT already in slot-filling flow.
    in_slot_flow = bool(state and state.get("current_intent") == "build_plan")
    if not in_slot_flow and _has_rich_context(memory):
        # Clear any stale slot-filling state
        memory.clear_action_state()
        plan = _generate_plan_from_context(user_input, memory, llm_fn)
        return {
            "type": "plan",
            "plan": plan,
            "message": _format_plan_for_display(plan),
        }

    # --- Path 2: Cold-start slot-filling ---
    if state and state.get("current_intent") == "build_plan":
        slots = dict(state.get("slots", {}))
        is_new_flow = False
    else:
        slots = {}
        is_new_flow = True

    # New cold-start plan request: avoid storing trigger text as the "issue".
    if is_new_flow:
        prefilled_issue = _extract_issue_from_plan_request(user_input)
        if prefilled_issue:
            slots["issue"] = prefilled_issue
        else:
            memory.set_action_state("build_plan", slots)
            return {
                "type": "follow_up",
                "message": _SLOT_QUESTIONS["issue"],
                "slots": slots,
            }

    # Fill the next empty required slot
    next_empty = _next_missing_slot(slots)

    if not is_new_flow and next_empty and user_input.strip():
        # The user is answering the question for the next empty slot
        slots[next_empty] = user_input.strip()

    # Check again after filling
    next_empty = _next_missing_slot(slots)

    if next_empty:
        # Still missing slots — save state and ask follow-up
        memory.set_action_state("build_plan", slots)
        return {
            "type": "follow_up",
            "message": _SLOT_QUESTIONS[next_empty],
            "slots": slots,
        }

    # All required slots filled → generate the plan
    memory.clear_action_state()
    plan = _generate_plan(slots, llm_fn)

    return {
        "type": "plan",
        "plan": plan,
        "message": _format_plan_for_display(plan),
    }


def _next_missing_slot(slots: dict[str, Any]) -> str | None:
    """Return the first required slot that is not yet filled."""
    for slot in REQUIRED_SLOTS:
        if not slots.get(slot):
            return slot
    return None


def _generate_plan_from_context(
    user_input: str,
    memory: SessionMemory,
    llm_fn: Callable[[str, str], str],
) -> dict[str, Any]:
    """
    Generate a plan using the full conversation history as context.
    Used when the user has already described their situation in detail.
    """
    conversation_history = memory.get_history_for_prompt(limit=20)
    profile_context = memory.get_profile_for_prompt()

    prompt = BUILD_PLAN_FROM_CONTEXT_PROMPT.format(
        conversation_history=conversation_history,
        profile_context=profile_context,
        user_message=user_input,
    )

    try:
        raw = llm_fn(
            "You are a relationship communication coach. Generate a conversation "
            "plan that is SPECIFIC to this person's situation. Use the exact "
            "details they shared. Do not use generic placeholders.",
            prompt,
        )
    except Exception as e:
        return {
            "opening_statement": "I wasn't able to generate a plan right now.",
            "talking_points": [],
            "validating_phrase": "",
            "boundary_phrase": "",
            "suggested_next_question": "",
            "error": str(e),
        }

    plan = _parse_plan_response(raw, {"source": "conversation_context"})
    return plan


def _generate_plan(
    slots: dict[str, Any],
    llm_fn: Callable[[str, str], str],
) -> dict[str, Any]:
    """Call the LLM to produce a structured conversation plan."""
    prompt = BUILD_PLAN_PROMPT.format(
        issue=slots.get("issue", ""),
        goal=slots.get("goal", ""),
        tone=slots.get("tone", ""),
        delivery_mode=slots.get("delivery_mode", "in person"),
    )

    try:
        raw = llm_fn(
            "You are a relationship communication coach. Generate a conversation "
            "plan with SPECIFIC, CONCRETE phrases the user could say. Do not use "
            "generic placeholders — write actual sentences for their situation.",
            prompt,
        )
    except Exception as e:
        return {
            "opening_statement": "I wasn't able to generate a plan right now.",
            "talking_points": [],
            "validating_phrase": "",
            "boundary_phrase": "",
            "suggested_next_question": "",
            "error": str(e),
        }

    # Parse the LLM response into a structured dict.
    plan = _parse_plan_response(raw, slots)
    return plan


def _strip_markdown(text: str) -> str:
    """Remove markdown bold/italic wrappers and stray quotes from a string."""
    import re
    # Strip leading/trailing whitespace first
    t = text.strip()
    # Remove wrapping bold/italic markers (**, *, __)
    t = re.sub(r'^\*{1,3}', '', t)
    t = re.sub(r'\*{1,3}$', '', t)
    t = re.sub(r'^_{1,2}', '', t)
    t = re.sub(r'_{1,2}$', '', t)
    # Strip leading list markers
    t = re.sub(r'^[-•]\s*', '', t)
    t = re.sub(r'^\d+[\.\)]\s*', '', t)
    # Strip wrapping quotes
    t = t.strip().strip('"').strip("'").strip()
    return t


def _parse_plan_response(raw: str, slots: dict[str, Any]) -> dict[str, Any]:
    """
    Best-effort parsing of the LLM's plan text into structured fields.
    If parsing fails, the raw text is returned as the opening statement.
    """
    plan: dict[str, Any] = {
        "slots": slots,
        "opening_statement": "",
        "talking_points": [],
        "validating_phrase": "",
        "boundary_phrase": "",
        "suggested_next_question": "",
        "raw_text": raw,
    }

    lines = raw.strip().split("\n")
    current_section = ""

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if not stripped:
            continue

        if "opening statement" in lower or "opener" in lower:
            current_section = "opening"
            if ":" in stripped:
                content = _strip_markdown(stripped.split(":", 1)[1])
                if content:
                    plan["opening_statement"] = content
            continue
        elif "talking point" in lower:
            current_section = "talking"
            if ":" in stripped:
                content = _strip_markdown(stripped.split(":", 1)[1])
                if content:
                    plan["talking_points"].append(content)
            continue
        elif "validat" in lower and ("phrase" in lower or "statement" in lower):
            current_section = "validating"
            if ":" in stripped:
                content = _strip_markdown(stripped.split(":", 1)[1])
                if content:
                    plan["validating_phrase"] = content
            continue
        elif "boundary" in lower and ("phrase" in lower or "statement" in lower):
            current_section = "boundary"
            if ":" in stripped:
                content = _strip_markdown(stripped.split(":", 1)[1])
                if content:
                    plan["boundary_phrase"] = content
            continue
        elif "next question" in lower or "follow-up" in lower or "suggested" in lower:
            current_section = "next_question"
            if ":" in stripped:
                content = _strip_markdown(stripped.split(":", 1)[1])
                if content:
                    plan["suggested_next_question"] = content
            continue

        # Accumulate content under current section
        content = _strip_markdown(stripped)

        if current_section == "opening" and not plan["opening_statement"] and content:
            plan["opening_statement"] = content
        elif current_section == "talking" and content:
            plan["talking_points"].append(content)
        elif current_section == "validating" and not plan["validating_phrase"] and content:
            plan["validating_phrase"] = content
        elif current_section == "boundary" and not plan["boundary_phrase"] and content:
            plan["boundary_phrase"] = content
        elif current_section == "next_question" and not plan["suggested_next_question"] and content:
            plan["suggested_next_question"] = content

    # Fallback: if parsing extracted nothing, put raw text as opening
    if not plan["opening_statement"] and not plan["talking_points"]:
        plan["opening_statement"] = raw.strip()

    return plan


def _is_empty_content(text: str) -> bool:
    """Check if a parsed field is effectively empty (just markdown artifacts)."""
    if not text:
        return True
    cleaned = text.strip().strip('*').strip('"').strip("'").strip()
    return len(cleaned) < 3


def _format_plan_for_display(plan: dict[str, Any]) -> str:
    """Format a plan dict as a readable Markdown string for the chat UI."""
    parts: list[str] = []
    parts.append("Here's your conversation plan:\n")

    if plan.get("opening_statement") and not _is_empty_content(plan["opening_statement"]):
        parts.append(f"**Opening Statement:**\n\"{plan['opening_statement']}\"\n")

    # Filter out empty talking points
    real_points = [tp for tp in plan.get("talking_points", []) if not _is_empty_content(tp)]
    if real_points:
        parts.append("**Talking Points:**")
        for i, tp in enumerate(real_points[:5], 1):
            parts.append(f"  {i}. {tp}")
        parts.append("")

    if plan.get("validating_phrase") and not _is_empty_content(plan["validating_phrase"]):
        parts.append(f"**Validating Phrase:**\n*\"{plan['validating_phrase']}\"*\n")

    if plan.get("boundary_phrase") and not _is_empty_content(plan["boundary_phrase"]):
        parts.append(f"**Boundary Phrase:**\n*\"{plan['boundary_phrase']}\"*\n")

    if plan.get("suggested_next_question") and not _is_empty_content(plan["suggested_next_question"]):
        parts.append(f"**Suggested Follow-up:**\n{plan['suggested_next_question']}\n")

    parts.append(
        "---\n*Tip: Say **\"save my plan\"** to keep this for later, "
        "or ask me to adjust any part of it.*"
    )

    return "\n".join(parts)


# ============================================================================
# B. Reflection Exercise (SINGLE-TURN)
# ============================================================================

def handle_reflection(
    user_input: str,
    memory: SessionMemory,
    llm_fn: Callable[[str, str], str],
) -> dict[str, Any]:
    """
    Generate a guided reflection exercise based on the user's situation.
    Single-turn: no slot-filling required.
    """
    history = memory.get_history_for_prompt(limit=6)
    profile_context = memory.get_profile_for_prompt()

    prompt = REFLECTION_PROMPT.format(
        user_message=user_input,
        conversation_context=history if history else "No prior context.",
        profile_context=profile_context,
    )

    try:
        raw = llm_fn(
            "You are a supportive relationship reflection guide. "
            "Generate thoughtful reflection prompts. Be warm but concise.",
            prompt,
        )
        return {
            "type": "reflection",
            "message": raw.strip(),
        }
    except Exception as e:
        return {
            "type": "reflection",
            "message": (
                "I wasn't able to generate reflection prompts right now. "
                "Could you try describing your situation again?"
            ),
            "error": str(e),
        }


# ============================================================================
# C. Save / Retrieve Plan (SINGLE-TURN)
# ============================================================================

def handle_save_plan(
    session_id: str,
    latest_plan: dict[str, Any] | None,
    label: str = "default",
) -> dict[str, Any]:
    """
    Save the most recently generated plan to SQLite.
    """
    if not latest_plan:
        return {
            "type": "save_result",
            "success": False,
            "message": (
                "There's no plan to save yet. Try saying "
                "**\"help me build a conversation plan\"** first!"
            ),
        }

    try:
        row_id = db_save_plan(session_id, latest_plan, label=label)
        return {
            "type": "save_result",
            "success": True,
            "message": (
                f"Your plan has been saved (ID: {row_id}). "
                "You can retrieve it anytime by saying **\"show my saved plan\"**."
            ),
        }
    except Exception as e:
        return {
            "type": "save_result",
            "success": False,
            "message": "Sorry, I couldn't save the plan right now. Please try again.",
            "error": str(e),
        }


def handle_retrieve_plan(session_id: str) -> dict[str, Any]:
    """
    Retrieve saved plans from SQLite and format for display.
    """
    try:
        plans = get_saved_plans(session_id)
    except Exception as e:
        return {
            "type": "retrieve_result",
            "success": False,
            "message": "Sorry, I couldn't retrieve your plans right now.",
            "error": str(e),
        }

    if not plans:
        return {
            "type": "retrieve_result",
            "success": False,
            "message": (
                "You don't have any saved plans yet. "
                "Try **\"help me build a conversation plan\"** to get started!"
            ),
        }

    # Format the plans for display
    parts: list[str] = [f"I found **{len(plans)}** saved plan(s):\n"]
    for i, p in enumerate(plans, 1):
        plan_data = p["plan"]
        label = p.get("label", "default")
        created = p.get("created_at", "unknown")
        parts.append(f"### Plan {i} — *{label}* (saved {created})")
        if plan_data.get("opening_statement"):
            parts.append(f"**Opening:** {plan_data['opening_statement']}")
        if plan_data.get("talking_points"):
            parts.append("**Talking Points:**")
            for j, tp in enumerate(plan_data["talking_points"][:5], 1):
                parts.append(f"  {j}. {tp}")
        parts.append("")

    return {
        "type": "retrieve_result",
        "success": True,
        "message": "\n".join(parts),
        "plans": plans,
    }
