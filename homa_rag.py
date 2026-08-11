import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv

from embedder import get_embedder, collection, sync_knowledge_base, get_pending_changes


load_dotenv()

CHROMA_DB_PATH = os.environ.get(
    "HOMA_CHROMA_DB_PATH",
    str(Path(__file__).resolve().parent / "chroma_db"),
)

OLLAMA_URL = os.environ.get(
    "HOMA_OLLAMA_URL", "http://localhost:11434/api/generate"
)

LOG_PATH = os.environ.get(
    "HOMA_LOG_PATH",
    str(Path(__file__).resolve().parent / "homa_log.jsonl"),
)

MAX_DISTANCE = 0.85
TOP_K = 2

# Identity / anti-deflection preamble. raw=True bypasses the Modelfile SYSTEM,
# so if we want it on this path we must inject it into the prompt ourselves.
# It is folded into the FIRST user turn (Gemma has no system role). This is a
# deliberate deviation from the pure trained RAG format; the trained identity
# signal (heavily up-weighted in the V3 corpus) is the primary defense, this is
# a runtime nudge on top. Set HOMA_SYSTEM_PROMPT="" to disable and A/B it.
DEFAULT_SYSTEM = (
    "You are Homa, an offline agricultural assistant for farmers in Nigeria, "
    "built by Fallback AI. You give practical, direct advice on crops, livestock, "
    "soil, pests, weather, and markets in English, Hausa, Igbo, and Yoruba, "
    "replying in the language the user uses."
)
SYSTEM_PROMPT = os.environ.get("HOMA_SYSTEM_PROMPT", DEFAULT_SYSTEM)

# How many prior (user, assistant) exchanges to carry as multi-turn context.
# Capped so the running conversation plus fresh RAG passages stays under num_ctx.
MAX_HISTORY_TURNS = int(os.environ.get("HOMA_MAX_HISTORY_TURNS", "4"))


def _embed_query(question):
    """Embed a user question as a query vector for RAG search."""
    return list(get_embedder().embed([f"query: {question}"]))[0]


def search_rag(question, top_k=TOP_K):
    """Query the ChromaDB store and return nearest matched documents."""
    embedding = _embed_query(question)
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"[warn] search failed for homa_knowledge_base: {e}")
        return []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    scored = []
    for doc, meta, dist in zip(docs, metas, dists):
        if dist is None or dist > MAX_DISTANCE:
            continue
        scored.append(
            {
                "text": doc,
                "source": (meta or {}).get("source", "unknown"),
                "distance": dist,
            }
        )
    return scored

# The CURRENT user turn matches the SFT-trained RAG format exactly:
# "Retrieved Passages: / Passage N: / Question:", nothing else. When there's no
# retrieved context, it's the plain question (non-RAG training format).
# Only the current turn carries passages; prior turns are stored as the plain
# question + answer so multi-turn context stays compact and we don't re-feed
# stale passages.
def format_user_content(question, context_docs):
    if context_docs:
        passages = "\n\n".join(
            f"Passage {i+1}:\n{doc['text']}" for i, doc in enumerate(context_docs)
        )
        return f"Retrieved Passages:\n\n{passages}\n\nQuestion:\n{question}"
    return question
# Builds the full raw prompt string ourselves (raw=True), so we keep exact
# control over the turn format while supporting multi-turn history and an
# optional system preamble folded into the first user turn.
def build_prompt(question, context_docs, history=None, system=SYSTEM_PROMPT):
    """Build the RAG prompt for the language model, including retrieved passages."""
    history = history or []
    turns = list(history) + [("user", format_user_content(question, context_docs))]

    parts = []
    for idx, (role, content) in enumerate(turns):
        if idx == 0 and system:
            content = f"{system}\n\n{content}"
        tag = "model" if role == "assistant" else "user"
        parts.append(f"<start_of_turn>{tag}\n{content}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def clean_response(answer: str) -> str:
    if not answer:
        return ""
    cleaned = re.sub(r"Retrieved Passages:\s*", "",
                     answer, flags=re.IGNORECASE)
    cleaned = re.sub(r"Passage \d+:\s*", "", cleaned, flags=re.IGNORECASE)
    
    leak_phrases = [
        "i am homa, created by fallback",
        "my instructions",
        "the retrieved passages",
        "the provided text",
    ]
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    clean = [s for s in sentences if not any(
        p in s.lower() for p in leak_phrases)]
    return " ".join(clean).strip()


def log_turn(question, context_docs, raw_answer, clean_answer, retrieval_s, generation_s, error=None):
    """Persist a single question/answer turn to the JSONL log file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieved": [
            {
                "source": d["source"],
                "distance": round(d["distance"], 4),
                "text": d["text"],
            }
            for d in context_docs
        ],
        "raw_response": raw_answer,
        "clean_response": clean_answer,
        "retrieval_seconds": round(retrieval_s, 3) if retrieval_s is not None else None,
        "generation_seconds": round(generation_s, 3) if generation_s is not None else None,
        "error": error,
    }
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[warn] failed to write log entry: {e}")


def ask_homa(question, history=None):
    """Ask the Homa RAG agent a question and return its cleaned answer."""
    t0 = time.perf_counter()
    context_docs = search_rag(question)
    t1 = time.perf_counter()
    prompt = build_prompt(question, context_docs, history=history)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "homa",
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1024,
                    "num_ctx": 4096,
                    "stop": ["<start_of_turn>", "<end_of_turn>"],
                },
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_msg = f"[Error] Couldn't reach the model at {OLLAMA_URL}: {e}"
        log_turn(question, context_docs, None,
                 error_msg, t1 - t0, None, error=str(e))
        return error_msg

    t2 = time.perf_counter()
    try:
        raw_answer = response.json().get("response", "").strip()
    except ValueError:
        error_msg = "[Error] Model server returned an unexpected (non-JSON) response."
        log_turn(question, context_docs, None, error_msg,
                 t1 - t0, t2 - t1, error="non-JSON response")
        return error_msg

    clean_answer = clean_response(raw_answer)
    log_turn(question, context_docs, raw_answer,
             clean_answer, t1 - t0, t2 - t1)
    return clean_answer


def maybe_sync_knowledge_base():
    """Check for knowledge base changes and ask for permission before syncing.

    Detection (get_pending_changes) is read-only and cheap. The actual
    embedding pass (sync_knowledge_base) only runs if the user opts in,
    since it re-embeds every new/changed file and can be expensive.
    """
    try:
        pending = get_pending_changes()
    except Exception as e:
        print(f"[warn] failed to check for knowledge base changes: {e}")
        return

    new_or_updated = pending.get("new_or_updated", [])
    removed = pending.get("removed", [])

    if not new_or_updated and not removed:
        print("Knowledge base is up to date. No sync needed.")
        return

    print("Knowledge base changes detected:")
    if new_or_updated:
        print(f"  New or changed files ({len(new_or_updated)}):")
        for relpath in new_or_updated:
            print(f"    - {relpath}")
    if removed:
        print(f"  Removed files ({len(removed)}):")
        for relpath in removed:
            print(f"    - {relpath}")

    answer = input(
        "Re-embed / update the index now? [y/N]: ").strip().lower()
    if answer != "y":
        print("Skipping sync; using existing embeddings.")
        return

    print("Syncing knowledge base...")
    try:
        sync_knowledge_base()
    except Exception as e:
        print(f"[warn] sync_knowledge_base failed: {e}")


if __name__ == "__main__":
    maybe_sync_knowledge_base()

    print("Homa RAG Agent ready!")
    print(f"Model endpoint: {OLLAMA_URL}")
    print(f"Logging turns to: {LOG_PATH}")
    print("Type your farming question. 'quit' to exit, 'reset' to clear history.\n")
    # Rolling multi-turn context: list of ("user", q) / ("assistant", a) tuples.
    # Only the current turn gets RAG passages; history holds plain Q/A.
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "exit"]:
            break
        if question.lower() in ["reset", "/reset", "clear"]:
            history = []
            print("\n[history cleared]\n")
            continue
        if not question:
            continue
        answer = ask_homa(question, history=history)
        print(f"\nHoma: {answer}\n")
        # Append this exchange, then trim to the last MAX_HISTORY_TURNS exchanges.
        history.append(("user", question))
        history.append(("assistant", answer))
        history = history[-2 * MAX_HISTORY_TURNS:]
