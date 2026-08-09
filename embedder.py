import os
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import fitz

_HERE = Path(__file__).resolve().parent
KNOWLEDGE_BASE = str(_HERE / "knowledge_base")
CHROMA_DB_PATH = str(_HERE / "chroma_db")

print("Loading embedding model...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded!")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

COLLECTIONS = {
    "disease_docs": "crop_disease",
    "fertilizer_docs": "fertilizer",
    "planting_docs": "planting",
    "pest_docs": "pest_management"
}

def extract_text(filepath):
    if filepath.endswith(".pdf"):
        doc = fitz.open(filepath)
        return " ".join(page.get_text() for page in doc)
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def chunk_text(text, size=400, overlap=50):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap) if words[i:i+size]]

def embed_folder(folder, collection_name):
    path = os.path.join(KNOWLEDGE_BASE, folder)
    if not os.path.exists(path):
        print(f"Skipping {folder} - not found")
        return
    files = [f for f in os.listdir(path) if f.endswith((".pdf", ".txt"))]
    if not files:
        print(f"No files in {folder}")
        return
    collection = client.get_or_create_collection(collection_name)
    print(f"\nEmbedding {len(files)} files from {folder}...")
    idx = 0
    for filename in files:
        print(f"  {filename}")
        text = extract_text(os.path.join(path, filename))
        for chunk in chunk_text(text):
            if len(chunk.strip()) > 50:
                collection.add(
                    documents=[chunk],
                    embeddings=[embedder.encode(chunk).tolist()],
                    ids=[f"{folder}_{idx}"],
                    metadatas=[{"source": filename}]
                )
                idx += 1
    print(f"  Done: {idx} chunks embedded")

for folder, collection in COLLECTIONS.items():
    embed_folder(folder, collection)

print("\nAll documents embedded!")
print(f"ChromaDB at: {CHROMA_DB_PATH}")
