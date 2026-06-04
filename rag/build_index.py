"""
================================================================================
Person B — Retrieval Lead
L.O.V.E. Relationship Support Agent | RSM 8430 Group 18
================================================================================
 
WHAT THIS SCRIPT DOES
---------------------
Reads Person A's filtered_counselchat.csv, embeds each document_text using the
course-provided embedding endpoint (OpenAI-compatible), and stores everything in
a persistent ChromaDB collection.
 
Run this ONCE to build the index. After that, use retriever.py to query it.
 
HOW TO RUN
----------
Step 1 — Install dependencies (only needed once):
    pip install chromadb pandas openai python-dotenv
 
Step 2 — Make sure .env is configured with EMBED_API_BASE, EMBED_API_KEY.
 
Step 3 — Run from the project root:
    python rag/build_index.py
 
Step 4 — Check the printed summary at the bottom.
    You should see 225 documents added (or however many are in the CSV).
    The index is saved to data/chroma_db/ and persists across restarts.
 
OUTPUT
------
data/chroma_db/   <-- ChromaDB persistent storage. Don't delete this.
"""
 
from __future__ import annotations
 
import json
import os
from pathlib import Path
 
import pandas as pd
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
 
# ================================================================================
# Config
# ================================================================================
 
CSV_PATH    = Path("data/filtered_counselchat.csv")
CHROMA_DIR  = Path("data/chroma_db")
COLLECTION  = "counselchat"
 
# Embedding endpoint (course-provided)
EMBED_API_BASE = os.environ.get("EMBED_API_BASE", "https://rsm-8430-a2.bjlkeng.io")
EMBED_API_KEY  = os.environ.get("EMBED_API_KEY", os.environ.get("LLM_API_KEY", ""))
EMBED_MODEL    = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

# Batch size for embedding API calls
EMBED_BATCH = 100

# ================================================================================
# Embedding function compatible with ChromaDB
# ================================================================================

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
        """ChromaDB calls this during add()."""
        return self._embed(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """ChromaDB 1.x calls this during add()."""
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """ChromaDB 1.x calls this during query()."""
        return self._embed(input)

# ================================================================================
# Main
# ================================================================================
 
def main():
    if not EMBED_API_KEY:
        print("ERROR: EMBED_API_KEY (or LLM_API_KEY) not set. Check your .env file.")
        return

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df)}")
 
    # ── Set up ChromaDB ───────────────────────────────────────────────────────
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
 
    ef = APIEmbeddingFunction(
        api_base=EMBED_API_BASE,
        api_key=EMBED_API_KEY,
        model=EMBED_MODEL,
    )
 
    # Delete and recreate so re-runs are idempotent.
    try:
        client.delete_collection(COLLECTION)
        print(f"  Deleted existing collection '{COLLECTION}'.")
    except Exception:
        pass
 
    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Created collection '{COLLECTION}'.")
 
    # ── Build documents, metadatas, ids ───────────────────────────────────────
    documents = df["document_text"].tolist()
    ids       = df["doc_id"].tolist()
 
    metadatas = df[[
        "doc_id", "questionID", "question_title",
        "project_topic", "original_topic", "tier", "upvotes", "ans_len"
    ]].rename(columns={"upvotes": "upvotes_num"}).to_dict(orient="records")
 
    # ChromaDB metadata values must be str/int/float/bool — coerce to be safe.
    for m in metadatas:
        for k, v in m.items():
            if not isinstance(v, (str, int, float, bool)):
                m[k] = str(v)
 
    # ── Add in batches (ChromaDB recommends <= 5000 per call) ─────────────────
    BATCH = 200
    total = len(documents)
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  Added {end}/{total} documents...")
 
    # ── Verify ────────────────────────────────────────────────────────────────
    count = collection.count()
    print()
    print("=" * 55)
    print("INDEX BUILD SUMMARY")
    print("=" * 55)
    print(f"  Documents in collection: {count}")
    if count == total:
        print(f"  ✅ All {total} documents indexed successfully.")
    else:
        print(f"  ❌ Mismatch — expected {total}, got {count}.")
 
    print(f"  Embedding endpoint: {EMBED_API_BASE}")
    print(f"  Embedding model:   {EMBED_MODEL}")
    print(f"  ChromaDB path:     {CHROMA_DIR}")
    print()
    print("Done. Now run: python rag/retriever.py")
 
 
if __name__ == "__main__":
    main()
 
