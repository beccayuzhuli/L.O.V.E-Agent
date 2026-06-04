"""
================================================================================
L.O.V.E. Relationship Support Agent — Safety & Guardrails
RSM 8430 Group 18
================================================================================

Screens every user message BEFORE intent routing. Returns a safety verdict:
  - "safe"         → proceed to routing
  - "crisis"       → emotional crisis / self-harm language
  - "abuse"        → abuse or violence signals
  - "medical"      → medical advice request
  - "legal"        → legal advice request
  - "injection"    → prompt injection attempt
  - "out_of_scope" → irrelevant to relationships

Design principles:
  - Conservative: flag obviously unsafe content, but do NOT over-trigger on
    normal relationship distress (e.g. "I'm so sad" is safe).
  - Simple keyword + pattern matching as first pass.
  - Does NOT roleplay a therapist or provide crisis hotline scripts.
  - Returns a category and a safe response the UI can display.

Usage:
    from agent.safety import screen_message
    result = screen_message(user_input)
    if result["safe"]:
        ... proceed to router ...
    else:
        ... display result["response"] ...
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Keyword / pattern lists
# ---------------------------------------------------------------------------

_CRISIS_PATTERNS: list[str] = [
    r"\bsuicid(e|al)\b",
    r"\bkill\s+(my|him|her|them)self\b",
    r"\bself[- ]?harm\b",
    r"\bwant\s+to\s+die\b",
    r"\bend\s+(my|it\s+all|everything)\b",
    r"\bnot\s+worth\s+living\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bcut(ting)?\s+myself\b",
    r"\boverdose\b",
]

_ABUSE_PATTERNS: list[str] = [
    r"\bhit(s|ting)?\s+me\b",
    r"\bbeat(s|ing)?\s+me\b",
    r"\bphysically\s+abus",
    r"\bsexually\s+abus",
    r"\bdomestic\s+violen",
    r"\bstalk(s|ing|er)\b",
    r"\bthreat(en|ened|ening)\s+(to\s+)?kill\b",
    r"\bforced\s+me\b",
    r"\brape[ds]?\b",
    r"\bstrangl",
    r"\bchok(e[ds]?|ing)\s+me\b",
]

_MEDICAL_PATTERNS: list[str] = [
    r"\bprescri(be|ption)\b",
    r"\bmedication\b",
    r"\bdiagnos(e|is)\b",
    r"\bantidepress",
    r"\bshould\s+i\s+take\b.*\b(pill|drug|medicine)\b",
    r"\bpsychiatric\s+medication\b",
]

_LEGAL_PATTERNS: list[str] = [
    r"\bcustody\s+(battle|hearing|lawyer|case)\b",
    r"\brestraining\s+order\b",
    r"\bdivorce\s+(lawyer|attorney|law)\b",
    r"\blegal\s+(advice|action|rights)\b",
    r"\bsue\s+(my|him|her)\b",
    r"\bcourt\s+order\b",
]

_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"you\s+are\s+now\s+(a|an)\b",
    r"system\s*:\s*",
    r"new\s+instructions?\s*:",
    r"forget\s+(everything|all|your)\s+(you|instructions|rules)",
    r"override\s+(your|the)\s+(instructions|system|rules)",
    r"pretend\s+you\s+are",
    r"act\s+as\s+(if\s+you\s+are|a)\b",
    r"jailbreak",
    r"DAN\s+mode",
    r"\bdo\s+anything\s+now\b",
]

_OUT_OF_SCOPE_PATTERNS: list[str] = [
    r"\b(stock|crypto|bitcoin|invest(ment)?|portfolio)\b",
    r"\b(recipe|cook(ing)?|bak(e|ing))\b",
    r"\bwrite\s+(me\s+)?(a\s+)?(code|program|essay|poem)\b",
    r"\b(weather|forecast)\b",
    r"\b(homework|math\s+problem|calculus)\b",
    r"\bwho\s+won\s+(the|last)\b",
    r"\btranslate\s+(this|the)\b",
]


# ---------------------------------------------------------------------------
# Safe response messages
# ---------------------------------------------------------------------------

_RESPONSES: dict[str, str] = {
    "crisis": (
        "It sounds like you may be going through something really serious. "
        "I'm not a licensed therapist and I'm not equipped to help in a crisis. "
        "Please reach out to a professional who can support you:\n\n"
        "• **988 Suicide & Crisis Lifeline** — call or text **988** (US)\n"
        "• **Crisis Text Line** — text **HOME** to **741741**\n\n"
        "You deserve real support from someone who can help."
    ),
    "abuse": (
        "What you're describing sounds like it could involve abuse or violence. "
        "Your safety matters most. I'm not able to provide the kind of help "
        "this situation needs.\n\n"
        "• **National Domestic Violence Hotline** — 1-800-799-7233\n"
        "• **thehotline.org** for online chat\n\n"
        "Please consider reaching out to someone who can help keep you safe."
    ),
    "medical": (
        "I'm not qualified to give medical or psychiatric advice. "
        "For questions about medication, diagnoses, or treatment, "
        "please consult a licensed healthcare provider. "
        "I'm here to help with relationship communication and reflection."
    ),
    "legal": (
        "I'm not able to provide legal advice. For questions about custody, "
        "divorce law, or legal rights, please consult a qualified attorney. "
        "I'm here to help with relationship communication and reflection."
    ),
    "injection": (
        "I appreciate the creative input, but I can only help with relationship "
        "support topics. If you have a question about communication, conflict, "
        "or navigating a relationship challenge, I'm happy to help!"
    ),
    "out_of_scope": (
        "That's outside what I can help with. I'm a relationship support agent "
        "— I help with things like navigating conflict, preparing for tough "
        "conversations, and reflecting on relationship dynamics.\n\n"
        "Feel free to ask me about any of those!"
    ),
}


# ---------------------------------------------------------------------------
# Screening function
# ---------------------------------------------------------------------------

def _check_patterns(text: str, patterns: list[str]) -> bool:
    """Return True if any pattern matches (case-insensitive)."""
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def screen_message(user_input: str) -> dict[str, Any]:
    """
    Screen a user message for safety issues.

    Returns:
        {
            "safe": bool,
            "category": str,      # "safe" | "crisis" | "abuse" | ...
            "response": str|None  # None if safe, otherwise the refusal message
        }
    """
    text = user_input.strip()

    if not text:
        return {"safe": False, "category": "out_of_scope",
                "response": "It looks like your message was empty. How can I help you today?"}

    # Order matters: crisis/abuse must take precedence over all other checks.
    checks: list[tuple[str, list[str]]] = [
        ("crisis",    _CRISIS_PATTERNS),
        ("abuse",     _ABUSE_PATTERNS),
        ("medical",   _MEDICAL_PATTERNS),
        ("legal",     _LEGAL_PATTERNS),
        ("injection", _INJECTION_PATTERNS),
        ("out_of_scope", _OUT_OF_SCOPE_PATTERNS),
    ]

    for category, patterns in checks:
        if _check_patterns(text, patterns):
            return {
                "safe": False,
                "category": category,
                "response": _RESPONSES[category],
            }

    return {"safe": True, "category": "safe", "response": None}
