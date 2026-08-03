import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = str(Path(__file__).resolve().parent / "chroma_db")

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

COLLECTIONS = {
    "crop_disease": client.get_or_create_collection("crop_disease"),
    "fertilizer": client.get_or_create_collection("fertilizer"),
    "planting": client.get_or_create_collection("planting"),
    "pest_management": client.get_or_create_collection("pest_management")
}

def search_rag(question, n_results=3):
    embedding = embedder.encode(question).tolist()
    all_results = []
    for name, collection in COLLECTIONS.items():
        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                all_results.append({"text": doc, "source": (meta or {}).get("source", "unknown"), "collection": name})
        except Exception as e:
            print(f"[warn] search failed for collection '{name}': {e}")
    return all_results[:3]

# Test it
question = "How do I treat yellowing maize leaves?"
print(f"Query: {question}\n")
results = search_rag(question)
for i, r in enumerate(results):
    print(f"Result {i+1} [{r['source']}]:")
    print(r["text"][:300])
    print()
