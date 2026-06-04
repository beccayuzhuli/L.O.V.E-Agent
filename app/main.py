"""
================================================================================
L.O.V.E. Relationship Support Agent — Streamlit App
RSM 8430 Group 18
================================================================================
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any project imports that read os.environ
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.actions import (
    handle_build_plan,
    handle_reflection,
    handle_retrieve_plan,
    handle_save_plan,
)
from agent.memory import SessionMemory
from agent.router import classify_intent
from agent.safety import screen_message
from app.llm_client import generate_text
from app.prompts import (
    CONVERSATION_READINESS_PROMPT,
    RAG_QUERY_REWRITE_PROMPT,
    RELATIONSHIP_TIP_CARD_PROMPT,
    RAG_SYNTHESIS_PROMPT,
    RAG_WEAK_MATCH_RESPONSE,
    SYSTEM_PROMPT,
)
from rag.formatting import format_citations, format_for_llm
from rag.retriever import get_retriever
from state.store import get_or_create_session, init_db, list_sessions

DISTANCE_THRESHOLD = 0.55
HYBRID_THRESHOLD = 0.18
TIP_CARD_MARKER = "\n\n---\n📚 "

SITUATION_PHRASES: dict[str, list[str]] = {
    "Trust": [
        "I found messages that made me question trust.",
        "My partner broke a promise and I feel guarded.",
        "I keep checking for reassurance and feel anxious.",
        "We said we'd move forward, but I still feel hurt.",
    ],
    "Conflict": [
        "We keep fighting about the same issue every week.",
        "Small disagreements escalate into personal attacks.",
        "One of us shuts down while the other pushes harder.",
        "We argue in circles and never resolve anything.",
    ],
    "Communication": [
        "I feel unheard when I try to explain my feelings.",
        "My message comes out harsh even when I mean well.",
        "Hard conversations end before we understand each other.",
        "I avoid topics because they always go badly.",
    ],
    "Boundaries": [
        "I said a limit, but it keeps getting crossed.",
        "I feel guilty when I try to say no.",
        "Family or friends are affecting our relationship boundaries.",
        "I need a respectful way to set non-negotiables.",
    ],
    "Breakup": [
        "I think we should break up but I feel conflicted.",
        "We already broke up, and I don't know how to cope.",
        "We still live together and need a breakup plan.",
        "I want to end things without causing more harm.",
    ],
    "Emotional Distance": [
        "We feel like roommates instead of partners.",
        "Affection and connection have dropped off lately.",
        "I miss closeness but don't know how to ask for it.",
        "We're polite, but it feels emotionally flat.",
    ],
}

SITUATION_FOLLOWUPS: dict[str, str] = {
    "Trust": (
        "Thank you for sharing this trust concern. Could you tell me what happened, "
        "when it happened, and what repair attempts have already been tried?"
    ),
    "Conflict": (
        "Thanks for naming this conflict pattern. What is the recurring trigger, "
        "how does escalation usually unfold, and what outcome do you want from the next conversation?"
    ),
    "Communication": (
        "I hear you. What exact moment recently felt most misunderstood, how did each of you respond, "
        "and what would feeling heard look like for you?"
    ),
    "Boundaries": (
        "Thanks for sharing this boundary situation. Which boundary is being crossed, "
        "how have you communicated it so far, and what consequence or next step feels fair to you?"
    ),
    "Breakup": (
        "I’m with you in this difficult breakup moment. What stage are you in right now, "
        "what constraints do you need to manage, and what support do you need most this week?"
    ),
    "Emotional Distance": (
        "Thank you for sharing this emotional distance concern. When did this shift begin, "
        "what signs of disconnection you notice most, and what kind of reconnection would feel meaningful?"
    ),
}

st.set_page_config(
    page_title="L.O.V.E. — Relationship Support Agent",
    page_icon="💬",
    layout="wide",
)


@st.cache_resource
def _init_once() -> None:
    init_db()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

            /* ---- Light mode (default) ---- */
            :root {
                --love-bg: #f7f1e8;
                --love-card: #fffaf3;
                --love-ink: #2a1d11;
                --love-soft: #7c5f46;
                --love-accent: #c86e3b;
                --love-accent-2: #2e6d63;
                --love-line: #e7d8c8;
                --love-sidebar-bg-start: #fff8f0;
                --love-sidebar-bg-end: #f4ede4;
                --love-hero-start: #fff8ef;
                --love-hero-mid: #fbe9d8;
                --love-hero-end: #e6f1ec;
                --love-radial-1: #f3d8bf;
                --love-radial-2: #cde2d9;
                --love-bg-top: #fbf7f2;
                --love-shadow: rgba(65, 42, 24, 0.08);
            }

            /* ---- Dark mode overrides ---- */
            @media (prefers-color-scheme: dark) {
                :root {
                    --love-bg: #1a1612;
                    --love-card: #23201b;
                    --love-ink: #ede4d8;
                    --love-soft: #b09a84;
                    --love-accent: #e08a52;
                    --love-accent-2: #5bb8a8;
                    --love-line: #3d342a;
                    --love-sidebar-bg-start: #1e1a15;
                    --love-sidebar-bg-end: #15120e;
                    --love-hero-start: #2a241d;
                    --love-hero-mid: #302518;
                    --love-hero-end: #1a2a25;
                    --love-radial-1: rgba(80, 55, 30, 0.35);
                    --love-radial-2: rgba(35, 75, 62, 0.30);
                    --love-bg-top: #1a1612;
                    --love-shadow: rgba(0, 0, 0, 0.30);
                }
            }

            /* Also respond to Streamlit's built-in theme data attribute */
            [data-theme="dark"] {
                --love-bg: #1a1612;
                --love-card: #23201b;
                --love-ink: #ede4d8;
                --love-soft: #b09a84;
                --love-accent: #e08a52;
                --love-accent-2: #5bb8a8;
                --love-line: #3d342a;
                --love-sidebar-bg-start: #1e1a15;
                --love-sidebar-bg-end: #15120e;
                --love-hero-start: #2a241d;
                --love-hero-mid: #302518;
                --love-hero-end: #1a2a25;
                --love-radial-1: rgba(80, 55, 30, 0.35);
                --love-radial-2: rgba(35, 75, 62, 0.30);
                --love-bg-top: #1a1612;
                --love-shadow: rgba(0, 0, 0, 0.30);
            }

            .stApp {
                background:
                    radial-gradient(circle at 15% 20%, var(--love-radial-1) 0%, transparent 34%),
                    radial-gradient(circle at 80% 15%, var(--love-radial-2) 0%, transparent 33%),
                    linear-gradient(180deg, var(--love-bg-top) 0%, var(--love-bg) 100%);
                color: var(--love-ink);
                font-family: "Source Sans 3", sans-serif;
            }

            h1, h2, h3 {
                font-family: "Fraunces", serif !important;
                color: var(--love-ink);
            }

            /* Ensure all text inherits the ink color */
            .stApp, .stApp p, .stApp span, .stApp li, .stApp label,
            .stApp .stMarkdown, .stApp .stText {
                color: var(--love-ink);
            }

            /* Chat message text */
            .stChatMessage, .stChatMessage p, .stChatMessage span {
                color: var(--love-ink) !important;
            }

            /* Sidebar */
            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, var(--love-sidebar-bg-start) 0%, var(--love-sidebar-bg-end) 100%);
                border-right: 1px solid var(--love-line);
            }

            div[data-testid="stSidebar"], div[data-testid="stSidebar"] p,
            div[data-testid="stSidebar"] span, div[data-testid="stSidebar"] label,
            div[data-testid="stSidebar"] .stMarkdown {
                color: var(--love-ink);
            }

            /* Sidebar buttons */
            div[data-testid="stSidebar"] button {
                color: var(--love-ink) !important;
                border-color: var(--love-line) !important;
            }

            /* Expander headers in sidebar */
            div[data-testid="stSidebar"] details summary span {
                color: var(--love-ink) !important;
            }

            .love-hero {
                background: linear-gradient(120deg, var(--love-hero-start) 0%, var(--love-hero-mid) 45%, var(--love-hero-end) 100%);
                border: 1px solid var(--love-line);
                border-radius: 20px;
                padding: 22px 24px;
                box-shadow: 0 12px 30px var(--love-shadow);
                margin-bottom: 8px;
                animation: fadeUp 320ms ease-out;
            }

            .love-hero h2 {
                margin: 0;
                font-size: 1.85rem;
                line-height: 1.2;
                color: var(--love-ink);
            }

            .love-hero p {
                margin: 10px 0 0;
                color: var(--love-soft);
                font-size: 1.05rem;
            }

            /* Header action buttons */
            .stApp button {
                color: var(--love-ink);
            }

            /* Chat input */
            .stChatInput, .stChatInput textarea {
                color: var(--love-ink) !important;
                background-color: var(--love-card) !important;
                border-color: var(--love-line) !important;
            }

            /* Selectbox / dropdowns */
            .stSelectbox div[data-baseweb="select"] {
                color: var(--love-ink);
                background-color: var(--love-card);
            }

            /* Expander content */
            details {
                background-color: var(--love-card) !important;
                border-color: var(--love-line) !important;
            }

            @keyframes fadeUp {
                from { transform: translateY(8px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_session() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = get_or_create_session()
    return st.session_state.session_id


def _load_messages_into_state(memory: SessionMemory) -> None:
    if "messages" not in st.session_state or not st.session_state.messages:
        history = memory.get_history(limit=150)
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]}
            for m in history
        ]
    if "latest_plan" not in st.session_state:
        st.session_state.latest_plan = None
    if "plan_offered" not in st.session_state:
        st.session_state.plan_offered = False
    if "pending_user_message" not in st.session_state:
        st.session_state.pending_user_message = ""


def _post_guided_situation_turn(memory: SessionMemory, situation: str, phrase: str) -> None:
    user_message = phrase.strip()
    follow_up = SITUATION_FOLLOWUPS.get(
        situation,
        "Thanks for sharing. Could you tell me what happened, what matters most to you, and what support you want next?",
    )

    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": follow_up})
    memory.add_user_message(user_message, intent="guided_situation")
    memory.add_assistant_message(follow_up, intent="guided_followup")
    st.session_state.pending_user_message = ""


def _render_sidebar(memory: SessionMemory) -> None:
    with st.sidebar:
        st.markdown("## L.O.V.E.")
        st.caption("Listen · Open Dialogue · Validate · Encourage")
        st.divider()

        sid = memory.session_id

        if st.button("Start New Session", use_container_width=True):
            st.session_state.session_id = get_or_create_session()
            st.session_state.messages = []
            st.session_state.latest_plan = None
            st.session_state.plan_offered = False
            st.session_state.pending_user_message = ""
            st.rerun()

        st.markdown("**Resume Session**")
        sessions = list_sessions()
        session_ids = [s["session_id"] for s in sessions]
        if session_ids:
            selected = st.selectbox(
                "Select session",
                session_ids,
                index=session_ids.index(sid) if sid in session_ids else 0,
                label_visibility="collapsed",
            )
            if selected != sid and st.button("Load Selected Session", use_container_width=True):
                st.session_state.session_id = selected
                st.session_state.messages = []
                st.session_state.latest_plan = None
                st.session_state.plan_offered = False
                st.session_state.pending_user_message = ""
                st.rerun()
        else:
            st.caption("No previous sessions yet.")

        st.divider()
        action_state = memory.get_action_state() or {}
        if action_state.get("current_intent") == "build_plan":
            slots = action_state.get("slots", {})
            progress = len([k for k in ["issue", "goal", "tone"] if slots.get(k)]) / 3
            st.markdown("**Plan Builder Progress**")
            st.progress(progress)
            st.caption(f"Current step: {len([k for k in ['issue', 'goal', 'tone'] if slots.get(k)]) + 1}/3")

        profile = memory.get_user_profile()
        if profile:
            st.markdown("**Profile Snapshot**")
            if profile.get("relationship_label"):
                st.caption(f"Relationship: {profile['relationship_label']}")
            if profile.get("focus_areas"):
                st.caption("Focus: " + ", ".join(profile["focus_areas"][:4]))
            if profile.get("preferred_tone"):
                st.caption(f"Tone: {profile['preferred_tone']}")

        st.divider()
        st.markdown("**Popular Situations**")
        for situation, phrases in SITUATION_PHRASES.items():
            with st.expander(situation, expanded=False):
                st.markdown(f"**{situation}**")
                for idx, phrase in enumerate(phrases):
                    if st.button(
                        phrase,
                        key=f"situation_{situation}_{idx}",
                        use_container_width=True,
                    ):
                        _post_guided_situation_turn(memory, situation, phrase)
                        st.rerun()

        st.divider()
        st.markdown(
            "**Quick Questions**\n"
            "- Help me build a conversation plan.\n"
            "- Save my plan.\n"
            "- Show my saved plan.\n"
            "- I want a reflection exercise."
        )
        st.caption(
            "This tool is for relationship support and communication practice, "
            "not a substitute for licensed professional care."
        )


def _render_header() -> None:
    st.markdown(
        """
        <div class="love-hero">
            <h2>L.O.V.E. Relationship Support Studio</h2>
            <p>Talk things through, build a clear conversation plan, and practice a healthier next step.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    options = [
        ("Talk Through A Fight", "my partner and i are always arguing and i need help"),
        ("Build A Plan", "help me build a conversation plan"),
        ("Reflection Exercise", "i want a reflection exercise"),
        ("Rebuild Trust", "i want to rebuild trust after betrayal"),
    ]
    cols = st.columns(4)
    for col, (label, message) in zip(cols, options):
        if col.button(label, use_container_width=True):
            st.session_state.pending_user_message = message


def _is_tip_card_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {"category", "actionable_examples", "source_summary"}
    return required.issubset(set(payload.keys()))


def _render_tip_card(payload: dict[str, object]) -> None:
    with st.expander("Your Relationship Tip Card"):
        category = str(payload.get("category", "Relationship")).strip() or "Relationship"
        tips = payload.get("actionable_examples", [])
        source_summary = str(payload.get("source_summary", "")).strip()

        st.markdown(f"**Category:** **{category}**")
        st.markdown("**Actionable examples:**")
        if isinstance(tips, list) and tips:
            for bullet in tips[:3]:
                text = str(bullet).strip()
                if text:
                    st.markdown(f"- {text}")
        else:
            st.markdown("- Use calm, specific language to describe what happened and how it affected you.")

        dataset_url = "https://huggingface.co/datasets/nbertagnolli/counsel-chat"
        themes = source_summary if source_summary else "relationship communication, conflict navigation"
        st.markdown(
            f"<small>Sourced therapist examples from "
            f"<a href='{dataset_url}' target='_blank'>CounselChat</a>: {themes}</small>",
            unsafe_allow_html=True,
        )


def _extract_legacy_tip_card(content: str) -> tuple[str, dict[str, object] | None]:
    """
    Backward-compatible parser for older fallback messages that ended with raw tip-card JSON.
    """
    match = re.match(r"^(?P<body>[\s\S]*?)\n\n(?P<json>\{[\s\S]+\})\s*$", content.strip())
    if not match:
        return content, None
    raw = match.group("json").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        return content, None
    if _is_tip_card_payload(parsed):
        return match.group("body").strip(), parsed
    return content, None


def _render_assistant_content(content: str) -> None:
    if TIP_CARD_MARKER in content:
        body, tip_card_raw = content.split(TIP_CARD_MARKER, 1)
        st.markdown(body)
        try:
            tip_card = json.loads(tip_card_raw.strip())
        except Exception:
            st.markdown(tip_card_raw)
            return

        if _is_tip_card_payload(tip_card):
            _render_tip_card(tip_card)
        else:
            st.markdown(tip_card_raw)
        return

    legacy_body, legacy_tip_card = _extract_legacy_tip_card(content)
    if legacy_tip_card:
        st.markdown(legacy_body)
        _render_tip_card(legacy_tip_card)
        return

    st.markdown(content)


def _build_example_summaries(results: list[dict]) -> str:
    """Build compact summaries for the UI tailoring prompt."""
    lines: list[str] = []
    for i, row in enumerate(results, 1):
        doc_id = row.get("doc_id", "unknown")
        topic = row.get("project_topic", "relationship")
        title = row.get("question_title", "")
        snippet = (row.get("answer_snippet") or row.get("answer_text") or "").replace("\n", " ").strip()
        if len(snippet) > 180:
            snippet = snippet[:180].rstrip() + "..."
        lines.append(
            f"Example {i}: id={doc_id}; topic={topic}; title={title}; therapist_answer_snippet={snippet}"
        )
    return "\n".join(lines)


def _generate_tip_card_actions(
    user_input: str,
    memory: SessionMemory,
    results: list[dict],
) -> list[str]:
    """Generate 1-3 concise actionable bullets for the relationship tip card."""
    if not results:
        return []

    history = memory.get_history_for_prompt(limit=6) or "No prior history."
    profile_context = memory.get_profile_for_prompt()
    example_summaries = _build_example_summaries(results)
    prompt = RELATIONSHIP_TIP_CARD_PROMPT.format(
        user_message=user_input,
        history=history,
        profile_context=profile_context,
        example_summaries=example_summaries,
    )

    try:
        raw = generate_text(
            (
                "You are a relationship coach. Output only complete, actionable suggestions "
                "in second person. Never describe the examples or summarize the situation."
            ),
            prompt,
            temperature=0.35,
            max_tokens=512,
        ).strip()
    except Exception:
        return [
            "Name what happened using one concrete example and your feeling.",
            "Ask one open-ended question to understand your partner's perspective.",
            "Agree on one small, time-bound next step to reduce repeated conflict.",
        ]

    cleaned: list[str] = []
    for line in raw.splitlines():
        t = line.strip().lstrip("-•*").strip()
        if not t:
            continue
        t = re.sub(r"^\d+[\.\)]\s*", "", t)
        t = t.strip()
        if not t:
            continue
        # Skip lines that are meta-commentary rather than advice
        lower_t = t.lower()
        if any(lower_t.startswith(prefix) for prefix in (
            "the user", "the example", "based on", "okay,", "so,",
            "the tip", "this example", "here are", "here is",
        )):
            continue
        cleaned.append(t)
        if len(cleaned) >= 3:
            break

    if not cleaned:
        return [
            "Name what happened using one concrete example and your feeling.",
            "Ask one open-ended question to understand your partner's perspective.",
            "Agree on one small, time-bound next step to reduce repeated conflict.",
        ]

    # De-duplicate while preserving order, and cap at 3 bullets.
    deduped: list[str] = []
    seen: set[str] = set()
    for item in cleaned:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= 3:
            break
    return deduped[:3]


def _llm_issue_hint(exc: Exception) -> str | None:
    """
    Convert low-level LLM/API errors into user-safe, actionable guidance.
    """
    message = str(exc).lower()
    if any(token in message for token in ("401", "403", "api key", "unauthorized", "forbidden")):
        return "Please check `LLM_API_KEY` for your current terminal session."
    if any(token in message for token in ("model", "not found", "does not exist")):
        return "Please verify `LLM_MODEL` matches an available model on your endpoint (e.g., `qwen3-30b-a3b-fp8`)."
    if any(token in message for token in ("connection", "failed to establish", "name or service not known", "timed out")):
        return "Please verify `LLM_API_BASE` and that your model endpoint is reachable."
    return None


def _display_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_content(msg["content"])
            else:
                st.markdown(msg["content"])


def _rewrite_query_for_retrieval(user_input: str, memory: SessionMemory) -> str:
    history = memory.get_history_for_prompt(limit=4) or "No prior history."
    profile_context = memory.get_profile_for_prompt()
    prompt = RAG_QUERY_REWRITE_PROMPT.format(
        history=history,
        profile_context=profile_context,
        user_message=user_input,
    )
    try:
        rewritten = generate_text(
            "You rewrite relationship-support user messages into retrieval queries.",
            prompt,
            temperature=0.2,
            max_tokens=80,
        ).strip()
    except Exception:
        return user_input

    rewritten = rewritten.splitlines()[0].strip().strip('"').strip("'")
    if 3 <= len(rewritten) <= 240:
        return rewritten
    return user_input


def _is_weak_retrieval(results: list[dict]) -> bool:
    if not results:
        return True
    top = results[0]
    hybrid = float(top.get("hybrid_score", 0.0))
    distance = float(top.get("distance", 1.0))
    return hybrid < HYBRID_THRESHOLD and distance > DISTANCE_THRESHOLD


def _handle_rag(user_input: str, memory: SessionMemory) -> str:
    retrieval_query = _rewrite_query_for_retrieval(user_input, memory)

    try:
        retriever = get_retriever()
        results = retriever.retrieve(retrieval_query, k=4)
    except Exception:
        return (
            "I'm having trouble accessing the knowledge base right now. "
            "Could you try again in a moment?"
        )

    if _is_weak_retrieval(results):
        weak_results = results[:1]
        if weak_results:
            tip_actions = _generate_tip_card_actions(user_input, memory, weak_results)
            tip_card = format_citations(
                weak_results,
                actionable_examples=tip_actions,
                max_examples=1,
            )
            return f"{RAG_WEAK_MATCH_RESPONSE}\n\n---\n📚 {json.dumps(tip_card, ensure_ascii=False)}"
        return RAG_WEAK_MATCH_RESPONSE

    top_results = results[:2]
    tip_actions = _generate_tip_card_actions(user_input, memory, top_results)
    context = format_for_llm(top_results)
    tip_card = format_citations(
        top_results,
        actionable_examples=tip_actions,
        max_examples=2,
    )
    history = memory.get_history_for_prompt(limit=8) or "No prior history."
    profile_context = memory.get_profile_for_prompt()

    prompt = RAG_SYNTHESIS_PROMPT.format(
        context=context,
        history=history,
        profile_context=profile_context,
        user_message=user_input,
    )

    try:
        answer = generate_text(SYSTEM_PROMPT, prompt).strip()
        if not answer:
            raise RuntimeError("Empty synthesis response.")
    except Exception as exc:
        hint = _llm_issue_hint(exc)
        guidance = f"\n\n{hint}" if hint else ""
        return (
            "I’m having trouble generating the full response right now, but I still found "
            "relevant therapist examples and practical next steps for you."
            + guidance
            + f"{TIP_CARD_MARKER}{json.dumps(tip_card, ensure_ascii=False)}"
        )

    return f"{answer}{TIP_CARD_MARKER}{json.dumps(tip_card, ensure_ascii=False)}"


# ---------------------------------------------------------------------------
# Conversation readiness detection
# ---------------------------------------------------------------------------

_READINESS_MIN_USER_MESSAGES = 3


def _check_conversation_readiness(user_input: str, memory: SessionMemory) -> bool:
    """Return True if the user seems ready to be offered a conversation plan."""
    history = memory.get_history(limit=50)
    user_msg_count = sum(1 for m in history if m["role"] == "user")
    if user_msg_count < _READINESS_MIN_USER_MESSAGES:
        return False

    history_text = memory.get_history_for_prompt(limit=10)
    prompt = CONVERSATION_READINESS_PROMPT.format(
        history=history_text,
        user_message=user_input,
    )
    try:
        result = generate_text(
            "You are an evaluator. Respond with ONLY one word.",
            prompt,
            temperature=0.1,
            max_tokens=16,
        ).strip().lower()
        return "ready" in result and "not_ready" not in result
    except Exception:
        return False


def _is_affirmative(text: str) -> bool:
    """Return True if the message is a short affirmative response."""
    lower = text.lower().strip().rstrip("!.")
    affirmatives = {
        "yes", "yeah", "yep", "yea", "sure", "ok", "okay", "absolutely",
        "definitely", "please", "let's do it", "let's go", "sounds good",
        "i'd like that", "that would be great", "yes please", "go ahead",
        "i'm ready", "let's try", "sounds great", "i think so",
        "that sounds helpful", "why not", "for sure",
    }
    if lower in affirmatives:
        return True
    # Also match short messages that contain affirmative words
    if len(lower.split()) <= 5 and any(w in lower for w in ("yes", "yeah", "sure", "ok", "okay", "please", "let's", "ready")):
        return True
    return False


def _process_message(user_input: str, memory: SessionMemory) -> str:
    session_id = memory.session_id

    safety = screen_message(user_input)
    if not safety["safe"]:
        memory.add_user_message(user_input, intent=safety["category"])
        response = safety["response"]
        memory.add_assistant_message(response, intent=safety["category"])
        return response

    memory.infer_and_update_profile(user_input)
    action_state = memory.get_action_state()
    intent, _ = classify_intent(user_input, action_state, generate_text)

    # If we previously offered a plan and the user seems to accept, route to build_plan
    if (
        intent == "rag_qa"
        and st.session_state.get("plan_offered")
        and _is_affirmative(user_input)
    ):
        intent = "build_plan"
        st.session_state.plan_offered = False

    memory.add_user_message(user_input, intent=intent)

    try:
        if intent == "rag_qa":
            response = _handle_rag(user_input, memory)
            # Check if user is ready for a plan suggestion
            if _check_conversation_readiness(user_input, memory):
                st.session_state.plan_offered = True
        elif intent == "build_plan":
            result = handle_build_plan(user_input, memory, generate_text)
            response = result["message"]
            if result["type"] == "plan":
                st.session_state.latest_plan = result["plan"]
        elif intent == "reflection":
            result = handle_reflection(user_input, memory, generate_text)
            response = result["message"]
        elif intent == "save_plan":
            result = handle_save_plan(session_id, st.session_state.get("latest_plan"))
            response = result["message"]
        elif intent == "retrieve_plan":
            result = handle_retrieve_plan(session_id)
            response = result["message"]
        elif intent == "out_of_scope":
            response = (
                "That’s outside what I can help with directly. I can still support you "
                "with relationship communication, conflict navigation, and reflection."
            )
        else:
            response = (
                "I want to make sure I support you well. Could you share a bit more "
                "about the relationship situation you’re navigating?"
            )
    except Exception as exc:
        hint = _llm_issue_hint(exc)
        if hint:
            response = (
                "I ran into an issue while processing that message.\n\n"
                f"{hint}"
            )
        else:
            response = (
                "I ran into an issue while processing that message. "
                "Could you try rephrasing it?"
            )

    memory.add_assistant_message(response, intent=intent)
    return response


def _submit_user_message(user_input: str, memory: SessionMemory) -> None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = _process_message(user_input, memory)
        _render_assistant_content(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    _init_once()
    _inject_styles()

    session_id = _ensure_session()
    memory = SessionMemory(session_id)
    _load_messages_into_state(memory)

    _render_sidebar(memory)
    _render_header()

    if not st.session_state.messages:
        welcome = (
            "Welcome, I’m **L.O.V.E.**.\n\n"
            "I’ll listen first, ask thoughtful follow-ups, validate what you’re feeling, "
            "and help you craft practical next steps.\n\n"
            "You can start by sharing what happened, or tap one of the guided actions above."
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome})
        memory.add_assistant_message(welcome, intent="system")

    _display_chat()

    pending = st.session_state.get("pending_user_message", "").strip()
    if pending:
        st.session_state.pending_user_message = ""
        _submit_user_message(pending, memory)
        st.rerun()

    typed = st.chat_input("Share what happened, or ask for a plan/reflection exercise...")
    if typed:
        _submit_user_message(typed, memory)


if __name__ == "__main__":
    main()
