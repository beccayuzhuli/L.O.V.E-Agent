"""
================================================================================
L.O.V.E. Relationship Support Agent — Intent Router
RSM 8430 Group 18
================================================================================

Classifies user messages into one of seven intents:
  rag_qa        — user is asking a relationship question (use retrieval)
  build_plan    — user wants to create a conversation plan
  reflection    — user wants a reflection / self-awareness exercise
  save_plan     — user wants to save a generated plan
  retrieve_plan — user wants to retrieve a previously saved plan
  unsafe        — flagged by safety (handled upstream, included for completeness)
  out_of_scope  — not relationship-related

Strategy:
  1. Keyword / heuristic rules for high-confidence intents (save, retrieve, plan,
     reflection) — fast and transparent.
  2. LLM fallback for ambiguous cases — uses a lightweight intent-classification
     prompt.

The router also respects in-progress multi-turn actions: if the user is in the
middle of build_plan slot-filling, new messages continue that flow unless the
user explicitly requests something else.

Usage:
    from agent.router import classify_intent
    intent, rationale = classify_intent(user_input, action_state, llm_fn)
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.prompts import INTENT_ROUTING_PROMPT


# ---------------------------------------------------------------------------
# Keyword-based rules (fast path)
# ---------------------------------------------------------------------------

def _keyword_classify(text: str) -> str | None:
    """
    Return an intent string if keyword rules match with high confidence,
    otherwise None (fall through to LLM).
    """
    lower = text.lower().strip()

    # --- save_plan ---
    if re.search(r"\bsave\s+(my\s+|the\s+|this\s+)?plan\b", lower):
        return "save_plan"
    if re.search(r"\bstore\s+(my\s+|the\s+|this\s+)?plan\b", lower):
        return "save_plan"

    # --- retrieve_plan ---
    if re.search(r"\b(retrieve|load|show|get|find|view)\s+(my\s+|the\s+)?(saved\s+|previous\s+|old\s+)?plan", lower):
        return "retrieve_plan"
    if re.search(r"\bprevious\s+plan\b", lower):
        return "retrieve_plan"
    if re.search(r"\bsaved\s+plan", lower):
        return "retrieve_plan"

    # --- build_plan (explicit request) ---
    if re.search(r"\b(build|create|make)\s+(a\s+|my\s+)?(conversation\s+)?plan\b", lower):
        return "build_plan"
    if re.search(r"\bhelp\s+me\s+(build|create|make|prepare|with)\s+(a\s+)?(conversation\s+)?plan\b", lower):
        return "build_plan"
    if re.search(r"\bprepare\s+(for\s+)?(a\s+)?(difficult|hard|tough)?\s*conversation\b", lower):
        return "build_plan"
    if re.search(r"\bplan\s+(a|my)\s+(conversation|talk|discussion)\b", lower):
        return "build_plan"
    if re.search(r"\bconversation\s+plan\b", lower):
        return "build_plan"
    # "adjust / customize / tailor the plan to my situation"
    if re.search(r"\b(adjust|customize|customise|tailor|personali[sz]e|update)\s+(it|the\s+plan|my\s+plan|this)", lower):
        return "build_plan"
    if re.search(r"\b(adjust|tailor|customize)\s+.*(my\s+situation|what\s+i\s+said|our\s+conversation)", lower):
        return "build_plan"

    # --- reflection ---
    if re.search(r"\breflect(ion)?\b", lower):
        return "reflection"
    if re.search(r"\bself[- ]?awareness\b", lower):
        return "reflection"
    if re.search(r"\bhelp\s+me\s+(think|understand)\s+(about\s+)?(my|how\s+i)\b", lower):
        return "reflection"

    return None


# ---------------------------------------------------------------------------
# LLM-based classification (fallback)
# ---------------------------------------------------------------------------

_VALID_INTENTS = {"rag_qa", "build_plan", "reflection", "save_plan",
                  "retrieve_plan", "unsafe", "out_of_scope"}


def _llm_classify(text: str,
                  llm_fn: Callable[[str, str], str]) -> tuple[str, str]:
    """
    Use the LLM to classify intent. Returns (intent, rationale).
    Falls back to rag_qa if the LLM returns something unexpected.
    """
    prompt = INTENT_ROUTING_PROMPT.format(user_message=text)
    try:
        raw = llm_fn(
            "You are an intent classifier. Respond ONLY with the intent label.",
            prompt,
        ).strip().lower()
    except Exception:
        return "rag_qa", "LLM classification failed; defaulting to rag_qa"

    # The LLM might return extra text; try to extract a valid intent.
    for intent in _VALID_INTENTS:
        if intent in raw:
            return intent, f"LLM classified as {intent}"

    # Default: treat as a general relationship question → rag_qa
    return "rag_qa", f"LLM returned '{raw}'; defaulting to rag_qa"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_intent(
    user_input: str,
    action_state: dict[str, Any] | None = None,
    llm_fn: Callable[[str, str], str] | None = None,
) -> tuple[str, str]:
    """
    Classify the user's message into an intent.

    Args:
        user_input:    The raw user message.
        action_state:  Current multi-turn action state (from memory), or None.
        llm_fn:        A function(system_prompt, user_prompt) -> str for LLM
                        classification. If None, only keyword rules are used.

    Returns:
        (intent, rationale) — the intent string and a short explanation.
    """
    text = user_input.strip()

    # 1. If there's an active multi-turn action, continue it UNLESS the user
    #    is clearly requesting something else.
    if action_state and action_state.get("current_intent") == "build_plan":
        # Check if the user is explicitly changing topics
        keyword_intent = _keyword_classify(text)
        if keyword_intent and keyword_intent != "build_plan":
            return keyword_intent, "User switched away from build_plan"
        # Otherwise, continue slot-filling
        return "build_plan", "Continuing multi-turn build_plan"

    # 2. Try keyword rules first
    keyword_intent = _keyword_classify(text)
    if keyword_intent:
        return keyword_intent, f"Keyword match → {keyword_intent}"

    # 3. LLM fallback
    if llm_fn:
        return _llm_classify(text, llm_fn)

    # 4. No LLM available — default to rag_qa
    return "rag_qa", "No keyword match and no LLM; defaulting to rag_qa"
