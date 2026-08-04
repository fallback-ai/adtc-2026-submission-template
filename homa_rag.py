import os
import re
import json
import time
from dotenv import load_dotenv
import requests
import chromadb
from pathlib import Path
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer


load_dotenv()

CHROMA_DB_PATH = os.environ.get(
    "HOMA_CHROMA_DB_PATH",
    str(Path(__file__).resolve().parent / "chroma_db"),
)


# Defaults to local Ollama, but can be overridden with an env var like a Kaggle/Colab
# session tunneled via ngrok for testing locally:
#   HOMA_OLLAMA_URL="https://xxxx.ngrok-free.app/api/generate" python homa_rag.py.
OLLAMA_URL = os.environ.get(
    "HOMA_OLLAMA_URL", "http://localhost:11434/api/generate"
)

# Performance monitoring and observability:
# JSONL, not a single .json array. Safe to append one line per turn without
# rewriting the whole file, and greppable/parseable line-by-line for review.
LOG_PATH = os.environ.get("HOMA_LOG_PATH", str(
    Path(__file__).resolve().parent / "homa_log.jsonl")
)

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# We'd have to add more data. Most especially for livestock domain
COLLECTIONS = {
    "crop_disease": client.get_or_create_collection("crop_disease"),
    "fertilizer": client.get_or_create_collection("fertilizer"),
    "planting": client.get_or_create_collection("planting"),
    "pest_management": client.get_or_create_collection("pest_management"),
}


# Semantic search across ALL collections directly, ranked by distance
# — no keyword pre-filter. The old keyword router (route_query) was English-only and
# silently returned zero context for Hausa/Igbo/Yoruba questions, since
# none of those words appear in a non-English question. This works
# regardless of question language because it relies entirely on the
# multilingual embedder, not English keyword matching.
# Prioritizing accuracy improvements with a slight, negligible performance trade-off.
def search_rag(question, n_results_per_collection=2, top_k=3):
    embedding = embedder.encode(question).tolist()
    scored = []
    for col_name, collection in COLLECTIONS.items():
        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=n_results_per_collection,
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                scored.append({
                    "text": doc,
                    "source": (meta or {}).get("source", "unknown"),
                    "distance": dist,
                })
        except Exception as e:
            print(f"[warn] search failed for collection '{col_name}': {e}")
    scored.sort(key=lambda r: r["distance"])
    return scored[:top_k]

# Matches the SFT-trained RAG format exactly: "Retrieved Passages: /
# Passage N: / Question:", nothing else in the user turn. The model wasn't
# trained to treat inline meta-instructions ("never mention
# documents", persona framing, etc.) as authoritative. It only processes
# them like any other text, not as directives. So adding them here
# doesn't reliably steer behavior and just risks the model commenting
# on them instead of following them. When there's no retrieved context,
# this sends the plain question, matching the non-RAG training format.


def build_prompt(question, context_docs):
    if context_docs:
        passages = "\n\n".join(
            f"Passage {i+1}:\n{r['text']}" for i, r in enumerate(context_docs)
        )
        user_content = f"Retrieved Passages:\n\n{passages}\n\nQuestion:\n{question}"
    else:
        user_content = question
    return f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"

# Lightweight safety net only, not the primary defense The actual fix
# for identity leaks is matching the trained prompt format
# exactly (as shown in build_prompt). Kept short and conservative to avoid
# false-positive stripping of legitimate advice sentences. Will Re-test
# whether this is even still needed once the corrected template is in use.


def clean_response(answer):
    leak_phrases = [
        "i am homa, created by fallback",
        "my instructions",
        "the retrieved passages",
        "the provided text",
        "i cannot"
    ]
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    clean = [s for s in sentences if not any(
        p in s.lower() for p in leak_phrases)]
    return " ".join(clean).strip()


# One JSON object per turn -- question, what was retrieved (source/distance/
# text), raw vs cleaned response, and timing, for manual performance review.
# Logs failures too (error field), not just successes.
def log_turn(question, context_docs, raw_answer, clean_answer, retrieval_s, generation_s, error=None):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieved": [
            {"source": d["source"], "distance": round(
                d["distance"], 4), "text": d["text"]}
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


def ask_homa(question):
    t0 = time.perf_counter()
    context_docs = search_rag(question)
    t1 = time.perf_counter()
    prompt = build_prompt(question, context_docs)

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
            timeout=300,  # Increased in colab session.
            # Timeout can be reduced when testing locally
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


if __name__ == "__main__":
    print("Homa RAG Agent ready!")
    print(f"Model endpoint: {OLLAMA_URL}")
    print(f"Logging turns to: {LOG_PATH}")
    print("Type your farming question. Type quit to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "exit"]:
            break
        if not question:
            continue
        answer = ask_homa(question)
        print(f"\nHoma: {answer}\n")
