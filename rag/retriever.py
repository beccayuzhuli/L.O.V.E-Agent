"""
================================================================================
L.O.V.E. Relationship Support Agent — Hybrid Retriever
RSM 8430 Group 18
================================================================================

Provides a hybrid retrieval strategy:
  1) Vector retrieval via ChromaDB embeddings
  2) Lightweight lexical retrieval via BM25-style scoring
  3) Score fusion and reranking for stronger relevance

The retriever still returns the original fields expected by the app, while
adding extra ranking diagnostics:
  - semantic_score
  - lexical_score
  - hybrid_score
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ================================================================================
# Config — must match build_index.py
# ================================================================================

CHROMA_DIR = Path("data/chroma_db")
COLLECTION = "counselchat"
DEFAULT_K = 3

# Embedding endpoint (course-provided)
EMBED_API_BASE = os.environ.get("EMBED_API_BASE", "https://rsm-8430-a2.bjlkeng.io")
EMBED_API_KEY  = os.environ.get("EMBED_API_KEY", os.environ.get("LLM_API_KEY", ""))
EMBED_MODEL    = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

EMBED_BATCH = 100


class APIEmbeddingFunction:
    """ChromaDB-compatible embedding function that calls an OpenAI-compatible endpoint."""

    def __init__(self, api_base: str, api_key: str, model: str):
        self._client = OpenAI(base_url=f"{api_base.rstrip('/')}/v1", api_key=api_key)
        self._model = model

    def name(self) -> str:
        return "api_embedding"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            response = self._client.embeddings.create(input=batch, model=self._model)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    """Simple lexical tokenizer for lightweight BM25 scoring."""
    return _TOKEN_PATTERN.findall(text.lower())


def _minmax_normalize(values: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1], handling degenerate ranges."""
    if not values:
        return {}

    v_min = min(values.values())
    v_max = max(values.values())
    if abs(v_max - v_min) < 1e-9:
        return {k: (1.0 if v_max > 0 else 0.0) for k in values}

    return {k: (v - v_min) / (v_max - v_min) for k, v in values.items()}


class CounselChatRetriever:
    """
    Hybrid retriever that fuses vector relevance and lexical relevance.
    Instantiate once and reuse.
    """

    def __init__(
        self,
        chroma_dir: str | Path = CHROMA_DIR,
        collection_name: str = COLLECTION,
        embed_api_base: str = EMBED_API_BASE,
        embed_api_key: str = EMBED_API_KEY,
        embed_model: str = EMBED_MODEL,
    ):
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._ef = APIEmbeddingFunction(
            api_base=embed_api_base,
            api_key=embed_api_key,
            model=embed_model,
        )
        self._collection = self._client.get_collection(
            name=collection_name,
            embedding_function=self._ef,
        )
        self._records: list[dict[str, Any]] = []
        self._records_by_id: dict[str, dict[str, Any]] = {}
        self._idf: dict[str, float] = {}
        self._avg_doc_len = 1.0
        self._build_lexical_index()

    def _build_lexical_index(self) -> None:
        """Load collection docs and build lexical stats for BM25-style scoring."""
        raw = self._collection.get(include=["documents", "metadatas"])
        ids = raw.get("ids", [])
        docs = raw.get("documents", [])
        metas = raw.get("metadatas", [])

        self._records = []
        df: Counter[str] = Counter()
        total_len = 0

        for doc_id, doc_text, meta in zip(ids, docs, metas):
            tokens = _tokenize(doc_text)
            tf = Counter(tokens)
            doc_len = len(tokens)
            total_len += doc_len
            for term in set(tokens):
                df[term] += 1

            record = {
                "doc_id": doc_id,
                "document_text": doc_text,
                "metadata": meta or {},
                "term_freq": tf,
                "doc_len": max(doc_len, 1),
            }
            self._records.append(record)

        n_docs = max(len(self._records), 1)
        self._avg_doc_len = max(total_len / n_docs, 1.0)

        self._idf = {}
        for term, freq in df.items():
            self._idf[term] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

        self._records_by_id = {r["doc_id"]: r for r in self._records}

    def _bm25_score(self, query_terms: list[str], record: dict[str, Any]) -> float:
        """Compute BM25-style lexical relevance for one document."""
        k1 = 1.5
        b = 0.75
        tf: Counter[str] = record["term_freq"]
        doc_len: int = record["doc_len"]
        score = 0.0

        for term in query_terms:
            term_tf = tf.get(term, 0)
            if term_tf == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denom = term_tf + k1 * (1 - b + b * doc_len / self._avg_doc_len)
            score += idf * ((term_tf * (k1 + 1)) / max(denom, 1e-6))

        return score

    def _lexical_candidates(
        self, query: str, limit: int, topic_filter: str | None = None
    ) -> list[tuple[str, float]]:
        """Return top lexical candidates as (doc_id, score)."""
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[str, float]] = []
        for record in self._records:
            topic = str(record["metadata"].get("project_topic", ""))
            if topic_filter and topic != topic_filter:
                continue
            score = self._bm25_score(query_terms, record)
            if score > 0:
                scored.append((record["doc_id"], score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _extract_fields(doc_text: str) -> tuple[str, str]:
        """Extract question_text and answer_text from document_text payload."""
        answer_text = ""
        if "Therapist Answer:" in doc_text:
            answer_text = doc_text.split("Therapist Answer:", 1)[1].strip()

        question_text = ""
        if "Question:" in doc_text:
            q_block = doc_text.split("Question:", 1)[1]
            question_text = q_block.split("Therapist Answer:")[0].strip()

        return question_text, answer_text

    def retrieve(
        self, query: str, k: int = DEFAULT_K, topic_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Returns top-k results using hybrid vector+lexical reranking.
        """
        if not query or not query.strip():
            return []

        total_docs = self._collection.count()
        if total_docs == 0:
            return []

        safe_k = max(1, min(k, total_docs))
        candidate_pool = min(max(safe_k * 4, safe_k), total_docs)
        where = {"project_topic": topic_filter} if topic_filter else None

        vector = self._collection.query(
            query_texts=[query.strip()],
            n_results=candidate_pool,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        candidate_map: dict[str, dict[str, Any]] = {}
        for doc_text, meta, dist in zip(
            vector["documents"][0], vector["metadatas"][0], vector["distances"][0]
        ):
            doc_id = str(meta.get("doc_id", ""))
            candidate_map[doc_id] = {
                "doc_id": doc_id,
                "document_text": doc_text,
                "metadata": meta,
                "distance": float(dist),
                "semantic_raw": max(0.0, 1.0 - float(dist)),
                "lexical_raw": 0.0,
            }

        lexical = self._lexical_candidates(query, limit=candidate_pool, topic_filter=topic_filter)
        for doc_id, lexical_score in lexical:
            record = self._records_by_id.get(doc_id)
            if not record:
                continue
            if doc_id not in candidate_map:
                candidate_map[doc_id] = {
                    "doc_id": doc_id,
                    "document_text": record["document_text"],
                    "metadata": record["metadata"],
                    "distance": 1.0,
                    "semantic_raw": 0.0,
                    "lexical_raw": float(lexical_score),
                }
            else:
                candidate_map[doc_id]["lexical_raw"] = float(lexical_score)

        semantic_vals = {doc_id: item["semantic_raw"] for doc_id, item in candidate_map.items()}
        lexical_vals = {doc_id: item["lexical_raw"] for doc_id, item in candidate_map.items()}

        semantic_norm = _minmax_normalize(semantic_vals)
        lexical_norm = _minmax_normalize(lexical_vals)

        ranked: list[dict[str, Any]] = []
        for doc_id, item in candidate_map.items():
            sem = semantic_norm.get(doc_id, 0.0)
            lex = lexical_norm.get(doc_id, 0.0)
            hybrid = 0.7 * sem + 0.3 * lex
            item["semantic_score"] = round(sem, 4)
            item["lexical_score"] = round(lex, 4)
            item["hybrid_score"] = round(hybrid, 4)
            ranked.append(item)

        ranked.sort(key=lambda x: (-x["hybrid_score"], x["distance"]))
        ranked = ranked[:safe_k]

        output: list[dict[str, Any]] = []
        for item in ranked:
            meta = item["metadata"] or {}
            doc_text = item["document_text"] or ""
            question_text, answer_text = self._extract_fields(doc_text)
            answer_snippet = answer_text[:300] + ("..." if len(answer_text) > 300 else "")

            title = str(meta.get("question_title", ""))
            topic = str(meta.get("project_topic", ""))
            doc_id = str(meta.get("doc_id", item["doc_id"]))
            citation = f"[{doc_id}] {title} ({topic})"

            output.append(
                {
                    "doc_id": doc_id,
                    "question_title": title,
                    "question_text": question_text,
                    "answer_text": answer_text,
                    "answer_snippet": answer_snippet,
                    "project_topic": topic,
                    "original_topic": str(meta.get("original_topic", "")),
                    "tier": int(meta.get("tier", 0)),
                    "distance": round(float(item["distance"]), 4),
                    "semantic_score": float(item["semantic_score"]),
                    "lexical_score": float(item["lexical_score"]),
                    "hybrid_score": float(item["hybrid_score"]),
                    "citation": citation,
                }
            )

        return output


_retriever: CounselChatRetriever | None = None


def get_retriever() -> CounselChatRetriever:
    """Returns a shared retriever instance (singleton)."""
    global _retriever
    if _retriever is None:
        _retriever = CounselChatRetriever()
    return _retriever


if __name__ == "__main__":
    retriever = get_retriever()
    demo_queries = [
        "my partner and I keep fighting",
        "how can I rebuild trust after cheating",
        "I need help preparing a hard relationship conversation",
    ]
    for query in demo_queries:
        print(f"\nQuery: {query}")
        for row in retriever.retrieve(query, k=3):
            print(
                f"- {row['doc_id']} | topic={row['project_topic']} | "
                f"hybrid={row['hybrid_score']} | dist={row['distance']}"
            )
