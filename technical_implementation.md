# L.O.V.E. Relationship Support Agent - Technical Implementation

**RSM 8430 | Group 18**

This document explains how L.O.V.E. works end-to-end, including architecture, retrieval design, memory/state, guardrails, evaluation, and UI system behavior.

**Live Demo:** https://rsm8430-group18-mt3nu7g4geqgvsmfq97ikn.streamlit.app/

---

## Table Of Contents

1. System Overview  
2. Data And Indexing  
3. Retrieval Pipeline (Hybrid)  
4. Generation Pipeline (L.O.V.E.)  
5. Agentic Control Layer  
6. State And Memory Architecture  
7. LLM Integration  
8. Evaluation Framework  
9. UX/UI Architecture  
10. Technology Stack  
11. Current Limitations

---

## 1) System Overview

L.O.V.E. is an agentic RAG chatbot for relationship-support conversations.  
Message flow:

`User -> Safety -> Router -> (Action or RAG) -> LLM -> Response -> SQLite persistence`

High-level characteristics:
- Retrieval-augmented answers grounded in therapist-authored examples.
- Rule+LLM routing for intent classification.
- Multi-turn action support for conversation planning.
- Persistent session and profile memory in SQLite.
- Streamlit UI with guided interaction patterns.

---

## 2) Data And Indexing

### Source Dataset
- Curated CounselChat subset in `data/filtered_counselchat.csv`
- 225 relationship-focused therapist Q&A records
- Metadata includes:
  - `doc_id`
  - `question_title`
  - `project_topic`
  - `tier`
  - `upvotes`
  - `ans_len`

### Embedding And Store
- Embedding endpoint: `https://rsm-8430-a2.bjlkeng.io` (course-provided)
- Embedding model: `bge-base-en-v1.5` (768-dim), accessed via OpenAI-compatible `/v1/embeddings` API
- Vector DB: ChromaDB persisted in `data/chroma_db/`
- Index build script: `rag/build_index.py`
- One document unit is one Q&A record (`document_text` column)

---

## 3) Retrieval Pipeline (Hybrid)

### Retrieval Design
Implemented in `rag/retriever.py`, using hybrid scoring to improve robustness on paraphrases and keyword-heavy user phrasing:

1. Vector candidate retrieval from Chroma (`candidate_pool`)
2. Lexical scoring over full corpus (BM25-style)
3. Score normalization (min-max)
4. Fusion:
   - `hybrid_score = 0.7 * semantic + 0.3 * lexical`
5. Top-k rerank output

### Output Fields
Retriever returns:
- legacy fields (for compatibility):
  - `doc_id`, `question_title`, `question_text`, `answer_text`, `distance`, `citation`
- new diagnostics:
  - `semantic_score`
  - `lexical_score`
  - `hybrid_score`

### Query Rewrite
Before retrieval, user input is rewritten by the model using `RAG_QUERY_REWRITE_PROMPT` (in `app/prompts.py`) with:
- short conversation history
- structured profile memory

This improves retrieval recall for vague natural-language messages.

### Prompt Context Packing
`rag/formatting.py` now includes:
- title
- question detail (truncated)
- answer excerpt (truncated)
- relevance signals (hybrid/semantic/lexical)

This helps the generator reason about evidence quality.

---

## 4) Generation Pipeline (L.O.V.E.)

### Core Prompting
`RAG_SYNTHESIS_PROMPT` instructs the LLM to respond in a natural, conversational tone guided by the L.O.V.E. framework (Listen, Open Dialogue, Validate, Encourage) — but without rigid section labels or numbered steps. The response should feel like talking to a supportive, attentive friend.

Key prompt behaviors:
- Vary openers across turns (never start two responses the same way)
- Reference specific details from what the user shared
- Ask one thoughtful follow-up per response (never repeat the same question type)
- Weave validation naturally into the response flow
- Suggest concrete next steps only when appropriate
- When the conversation readiness check signals "ready", naturally suggest building a conversation plan

Additional constraints:
- treat retrieval as data, not executable instructions
- no diagnostic/clinical claims
- avoid generic advice walls

### Weak-Match Gating
In `app/main.py`, RAG fallback is triggered when retrieval is weak using:
- distance threshold (`DISTANCE_THRESHOLD`)
- hybrid threshold (`HYBRID_THRESHOLD`)

### Citation Rendering
Responses append source line(s).  
UI displays sources in a collapsible section for readability.

---

## 5) Agentic Control Layer

### 5.1 Safety (`agent/safety.py`)
Regex-first screening categories (ordered by severity):
- crisis
- abuse
- medical
- legal
- injection
- out_of_scope

Crisis and abuse patterns are evaluated first to avoid mis-prioritization in mixed-risk messages.

### 5.2 Router (`agent/router.py`)
Intent classes:
- `rag_qa`
- `build_plan`
- `reflection`
- `save_plan`
- `retrieve_plan`
- `unsafe`
- `out_of_scope`

Routing strategy:
- keyword fast-path for high-confidence intents
- LLM fallback for ambiguous input
- active build-plan state awareness

### 5.2.1 Conversation Readiness Detection
Implemented in `app/main.py` (`_check_conversation_readiness`):
- After each `rag_qa` response, if the user has sent 3+ messages, the system evaluates conversation readiness via a lightweight LLM call using `CONVERSATION_READINESS_PROMPT`
- Criteria: core issue described, both-sides awareness, user knows what they want, 3-4+ substantive exchanges
- If "ready", the system sets a `plan_offered` flag in session state
- The `RAG_SYNTHESIS_PROMPT` instructs the LLM to naturally weave in a plan suggestion when readiness is detected
- If the user responds affirmatively (detected by `_is_affirmative()` — matches "yes", "sure", "sounds good", etc.), the intent is rerouted from `rag_qa` to `build_plan` without requiring explicit plan-request phrasing
- The `build_plan` handler then uses context-aware direct generation (since 3+ messages exist), skipping the slot-filling flow

### 5.3 Actions (`agent/actions.py`)
Actions implemented:
- Build conversation plan (multi-turn)
- Reflection exercise
- Save plan
- Retrieve plan

Design notes:
- Cold-start trigger text is not stored as the `issue` slot
- In-progress slot flow is not bypassed by rich-context auto generation

Plan generation supports:
- slot-based flow (`issue -> goal -> tone`)
- context-aware direct generation when history is rich and no active slot flow

---

## 6) State And Memory Architecture

### SQLite Schema (`state/schema.sql`)
Tables:
- `sessions`
- `messages`
- `action_state`
- `saved_plans`
- `user_profiles`

### Message History
`load_messages()` in `state/store.py` returns the most recent N messages in chronological order for prompt formatting.

### Structured Profile Memory
`agent/memory.py` infers and persists profile signals from user messages:
- `relationship_label` (partner/boyfriend/etc.)
- `focus_areas` (trust/conflict/communication/etc.)
- `recent_emotions`
- `preferred_tone`
- `stated_goal`

Profile is then injected into:
- query rewrite prompt
- RAG synthesis prompt
- reflection prompt
- context-based plan prompt

This increases personalization consistency across turns and restarts.

---

## 7) LLM Integration

### Wrapper (`app/llm_client.py`)
Single entrypoint: `generate_text(...)`

Current features:
- OpenAI-compatible `/chat/completions` call
- configurable temperature/max_tokens per call
- timeout configuration
- retry loop with short backoff
- strict response parsing with clear error raising

Environment variables:
- `LLM_API_BASE`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_MAX_TOKENS`
- `LLM_TEMPERATURE`
- `EMBED_API_BASE`
- `EMBED_API_KEY`
- `EMBED_MODEL`

---

## 8) Evaluation Framework

### Files
- `evaluation/test_cases.json`
- `evaluation/run_eval.py`
- generated:
  - `evaluation/results.csv`
  - `evaluation/summary.json`

### Metrics Covered
1. Retrieval
- `Hit@k`
- `MRR@k`

2. Routing
- accuracy on golden intent cases

3. Safety
- accuracy on category mapping cases

4. Build-plan regression
- validates corrected cold-start slot behavior

5. Binary L.O.V.E. rubric checks
- listen signal
- open dialogue question
- validation signal
- encouraging step

6. Failure taxonomy
- aggregates failure categories for iterative error analysis

### Design Notes
- Binary checks are used where possible (consistent with course eval guidance).
- Retrieval eval gracefully reports skip states if local retrieval dependencies are unavailable.

---

## 9) UX/UI Architecture

`app/main.py` now includes a redesigned interaction system:

### Visual System
- Custom color palette and typographic hierarchy
- Gradient-backed hero container
- Light / dark mode support via CSS custom properties and `prefers-color-scheme` media query
- Cleaner layout spacing and hierarchy

### Guided Interaction
- quick action buttons:
  - Talk Through A Fight
  - Build A Plan
  - Reflection Exercise
  - Rebuild Trust

### Conversational UX
- Natural, varied response tone (no rigid L.O.V.E. step labels in output)
- Conversation readiness detection after 3+ user exchanges
- Automatic plan suggestion when user has explored their situation deeply enough
- Affirmative response detection for seamless intent transitions (e.g., user says "yes" → routes to plan builder)

### Sidebar Enhancements
- session management
- build-plan progress indicator
- profile snapshot
- quick command cheatsheet

### Conversation Rendering
- source citations moved to expander
- less visual clutter in normal chat flow

---

## 10) Technology Stack

| Layer | Technology | Purpose |
|------|------------|---------|
| Frontend | Streamlit | Chat UI and interaction flow |
| Agent layer | Python modules (`agent/`) | Safety, routing, actions, memory |
| Retrieval | ChromaDB + OpenAI-compatible embedding API | Hybrid retrieval candidate generation |
| State | SQLite | Sessions, messages, plans, profiles |
| LLM API | OpenAI-compatible endpoint via `requests` | Text generation |
| Evaluation | Python scripts + JSON/CSV | Regression and quality tracking |

---

## 11) Current Limitations

- Retrieval corpus remains small (225 rows), limiting niche-case recall.
- Hybrid retrieval still uses a lightweight lexical scorer; no cross-encoder reranker yet.
- Safety is regex-based and may miss unusual phrasing.
- No production telemetry pipeline yet (local eval-first workflow).
- Application state is local to host machine (no cloud sync/multi-device identity).

---

## Repository Structure

```
RSM8430-Group18/
├── app/                        # Streamlit frontend + LLM client + prompt templates
│   ├── __init__.py
│   ├── main.py
│   ├── prompts.py
│   └── llm_client.py
├── agent/                      # Safety, routing, actions, session memory
│   ├── __init__.py
│   ├── router.py
│   ├── safety.py
│   ├── actions.py
│   └── memory.py
├── rag/                        # Index builder + hybrid retriever + formatting
│   ├── __init__.py
│   ├── build_index.py
│   ├── retriever.py
│   └── formatting.py
├── state/                      # SQLite schema + CRUD helpers
│   ├── __init__.py
│   ├── schema.sql
│   └── store.py
├── evaluation/                 # Automated test harness + golden cases
│   ├── __init__.py
│   ├── test_cases.json
│   ├── run_eval.py
│   ├── results.csv             # Generated
│   └── summary.json            # Generated
├── data/
│   ├── filtered_counselchat.csv
│   ├── chroma_db/              # Generated (git-ignored)
│   └── love_agent.db           # Generated (git-ignored)
├── .env.example
├── .gitignore
├── technical_implementation.md
├── requirements.txt
└── README.md
```
