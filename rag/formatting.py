"""
================================================================================
Person B — Retrieval Lead
L.O.V.E. Relationship Support Agent | RSM 8430 Group 18
================================================================================
 
WHAT THIS FILE DOES
-------------------
Formats retrieval results into strings the Synthesis/Response agent can use.
Person C/D call format_for_llm() to get a clean context block for the prompt.
"""
 
from __future__ import annotations


def _truncate(text: str, max_chars: int) -> str:
    clean = (text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def format_for_llm(
    results: list[dict],
    max_question_chars: int = 260,
    max_answer_chars: int = 900,
) -> str:
    """
    Formats retrieval results into a context block for the LLM prompt.
 
    Args:
        results:          List of result dicts from retriever.retrieve()
        max_chars_per_doc: Max chars to include per answer (avoids prompt bloat)
 
    Returns:
        A formatted string like:
 
            --- Source 1 [cc_0042] (marriage) ---
            Question: My husband wants a divorce...
            Therapist Answer: Wow that is tough...
            ---
 
    Usage in prompt:
        context = format_for_llm(results)
        prompt  = f"Based on these examples from therapists:\\n\\n{context}\\n\\nUser: {query}"
    """
    if not results:
        return "No relevant examples found in the knowledge base."
 
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        answer = _truncate(r.get("answer_text", ""), max_answer_chars)
        question = _truncate(
            r.get("question_text") or r.get("question_title", ""),
            max_question_chars,
        )
        semantic = r.get("semantic_score", 0.0)
        lexical = r.get("lexical_score", 0.0)
        hybrid = r.get("hybrid_score", 0.0)

        lines.append(f"--- Source {i} [{r['doc_id']}] ({r['project_topic']}) ---")
        lines.append(f"Question Title: {r['question_title']}")
        lines.append(f"Question Detail: {question}")
        lines.append(f"Therapist Answer: {answer}")
        lines.append(
            "Relevance Signals: "
            f"hybrid={hybrid:.2f}, semantic={semantic:.2f}, lexical={lexical:.2f}"
        )
        lines.append("---")
        lines.append("")
 
    return "\n".join(lines).strip()


def _humanize_topic(topic: str) -> str:
    """
    Convert internal topic labels into user-friendly display labels.
    Example: relationship_general -> Relationship (General)
    """
    normalized = (topic or "").strip().lower()
    custom = {
        "relationship_general": "Relationship (General)",
        "emotional_distance": "Emotional Distance",
    }
    if normalized in custom:
        return custom[normalized]
    if not normalized:
        return "Relationship"
    return normalized.replace("_", " ").title()
 
 
def format_citations(
    results: list[dict],
    actionable_examples: list[str] | None = None,
    source_summary: str | None = None,
    max_examples: int = 2,
) -> dict[str, object]:
    """
    Returns a structured tip-card payload for UI-controlled rendering.
    """
    if not results:
        return {
            "category": "Relationship",
            "actionable_examples": [],
            "source_summary": "",
        }

    limited = results[:max_examples]
    top_topic = limited[0].get("project_topic", "")
    category_label = _humanize_topic(top_topic)

    cleaned_examples: list[str] = []
    for item in actionable_examples or []:
        t = (item or "").strip()
        if not t:
            continue
        cleaned_examples.append(t)
        if len(cleaned_examples) >= 3:
            break

    if not cleaned_examples:
        cleaned_examples = [
            "Use calm, specific language to describe what happened and how it affected you.",
            "Ask one open-ended question so your partner can share their perspective.",
            "Propose one small next step you can both try before your next check-in.",
        ]

    summary = (source_summary or "").strip()
    if not summary:
        top_titles = [str(r.get("question_title", "")).strip() for r in limited]
        top_titles = [t for t in top_titles if t]
        # Produce just the theme titles so the UI layer can format them freely.
        summary = ", ".join(top_titles[:2]) if top_titles else ""

    return {
        "category": category_label,
        "actionable_examples": cleaned_examples[:3],
        "source_summary": summary,
    }
