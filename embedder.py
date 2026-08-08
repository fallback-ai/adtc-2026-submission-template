import hashlib
import os
import re
import subprocess
from pathlib import Path

import chromadb
import fitz
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

_HERE = Path(__file__).resolve().parent
KNOWLEDGE_BASE = str(_HERE / "knowledge_base")
CHROMA_DB_PATH = str(_HERE / "chroma_db")
BATCH_SIZE = 64
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
FOLDER_CATEGORY_MAP = {
    "disease_docs": "crop_disease",
    "fertilizer_docs": "fertilizer",
    "planting_docs": "planting",
    "pest_docs": "pest_management",
    "animal_husbandry_docs": "animal_husbandry",
    "cash_crops_processing_and_production_docs": "cash_crops_processing_and_production"
}

_EMBEDDER_INSTANCE = None


class AfroXLMRMiniEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./afro_mini_onnx_int8")
        self.session = ort.InferenceSession(
            "./afro_mini_onnx_int8/model.onnx",
            providers=["CPUExecutionProvider"],
        )

    def _clean_texts(self, texts):
        cleaned = []
        for text in texts:
            text = text.replace("passage: ", "").replace("query: ", "")
            cleaned.append(text.strip())
        return cleaned

    def embed(self, texts):
        if not texts:
            return []
        texts = self._clean_texts(texts)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        ort_inputs = {
            self.session.get_inputs()[0].name: input_ids,
            self.session.get_inputs()[1].name: attention_mask,
        }
        outputs = self.session.run(None, ort_inputs)
        last_hidden_state = outputs[0]

        mask = attention_mask.astype(np.float32)
        mask_sum = np.clip(mask.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)
        pooled = (last_hidden_state * mask[:, :, None]).sum(axis=1) / mask_sum
        normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return normalized.tolist()


def get_embedder():
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        if not Path("./afro_mini_onnx_int8/model.onnx").exists():
            print("Model bundle not found. Building INT8 ONNX model for Davlan/afro-xlmr-mini...")
            try:
                subprocess.run(
                    "optimum-cli export onnx --model Davlan/afro-xlmr-mini --task feature-extraction --optimize O4 ./afro_mini_onnx_int8",
                    shell=True,
                    check=True
                )
            except Exception as e:
                print(f"[error] optimum-cli failed or is missing: {e}")

        print("Loading AfroXLMR-Mini ONNX embedder...")
        _EMBEDDER_INSTANCE = AfroXLMRMiniEmbedder()
        print("Model loaded successfully!")
    return _EMBEDDER_INSTANCE

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = client.get_or_create_collection("homa_knowledge_base")
    # Probe check to catch vector dimension mismatch (1024 vs 384) early
    if collection.count() > 0:
        sample = collection.get(limit=1, include=["embeddings"])
        if sample.get("embeddings") is not None and len(sample["embeddings"]) > 0:
            if len(sample["embeddings"][0]) != 384:
                raise ValueError("Dimension mismatch detected in ChromaDB")
except Exception as e:
    print(f"[info] Resetting ChromaDB collection due to model schema update: {e}")
    try:
        client.delete_collection("homa_knowledge_base")
    except Exception:
        pass
    collection = client.create_collection("homa_knowledge_base")


def clean_text(text):
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def extract_text(filepath):
    filepath = str(filepath)
    if filepath.lower().endswith(".pdf"):
        pages = []
        with fitz.open(filepath) as doc:
            for page_num in range(len(doc)):
                raw = doc[page_num].get_text("text")
                cleaned = clean_text(raw)
                if len(cleaned.strip()) >= 40:
                    pages.append((page_num, cleaned))
        return pages

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        cleaned = clean_text(f.read())
        return [(0, cleaned)] if cleaned else []


def _chunk_block(block, max_chars, overlap):
    if len(block) <= max_chars:
        return [block]

    chunks = []
    cursor = 0
    while cursor < len(block):
        end = min(cursor + max_chars, len(block))
        window = block[cursor:end]
        if end < len(block):
            boundary = max(
                window.rfind(". "),
                window.rfind("? "),
                window.rfind("! "),
            )
            if boundary > int(max_chars * 0.5):
                end = cursor + boundary + 1
                window = block[cursor:end]
        chunks.append(window.strip())
        cursor = max(end - overlap, cursor + 1)
    return chunks


def chunk_text(text, max_chars=MAX_CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        chunks.extend(_chunk_block(paragraph, max_chars, overlap))

    merged = []
    current = ""
    for chunk in chunks:
        candidate = f"{current}\n\n{chunk}" if current else chunk
        if current and len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = chunk
    if current:
        merged.append(current)
    return [chunk.strip() for chunk in merged if chunk.strip()]


def _normalize_source(source_path):
    return source_path.replace("\\", "/")


def _chunk_id(relpath, page_num, chunk_idx, chunk_text):
    digest = hashlib.md5(
        f"{relpath}|{page_num}|{chunk_idx}|{chunk_text[:40]}".encode("utf-8")
    )
    return digest.hexdigest()


def _load_existing_files(collection):
    existing = {}
    try:
        results = collection.get(include=["metadatas", "ids"])
    except Exception:
        return existing

    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    for meta, id_ in zip(metadatas, ids):
        if not meta:
            continue
        source = meta.get("source")
        if not source:
            continue
        if source not in existing:
            existing[source] = {"mtime": None, "ids": []}
        existing[source]["ids"].append(id_)
        if existing[source]["mtime"] is None and meta.get("file_mtime") is not None:
            try:
                existing[source]["mtime"] = float(meta.get("file_mtime"))
            except (TypeError, ValueError):
                existing[source]["mtime"] = None
    return existing


def sync_knowledge_base():
    existing_files = _load_existing_files(collection)
    updated_files = 0
    skipped_files = 0
    written_chunks = 0

    for folder, category in FOLDER_CATEGORY_MAP.items():
        folder_path = Path(KNOWLEDGE_BASE) / folder
        if not folder_path.exists():
            print(f"Skipping missing folder: {folder}")
            continue

        for root, _, files in os.walk(folder_path):
            for filename in sorted(files):
                filepath = Path(root) / filename
                if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                relpath = _normalize_source(
                    str(filepath.relative_to(KNOWLEDGE_BASE)))
                mtime = os.path.getmtime(filepath)
                existing = existing_files.get(relpath)
                if existing and existing.get("mtime") == mtime:
                    skipped_files += 1
                    continue

                pages = extract_text(filepath)
                if not pages:
                    continue

                if existing and existing.get("ids"):
                    try:
                        collection.delete(ids=existing["ids"])
                    except Exception:
                        pass

                texts = []
                ids = []
                metadatas = []
                for page_num, page_text in pages:
                    page_chunks = chunk_text(page_text)
                    for chunk_idx, chunk in enumerate(page_chunks):
                        if len(chunk.strip()) < 50:
                            continue
                        ids.append(
                            _chunk_id(relpath, page_num, chunk_idx, chunk))
                        texts.append(f"passage: {chunk}")
                        metadatas.append({
                            "source": relpath,
                            "category": category,
                            "page": page_num,
                            "file_mtime": mtime,
                        })

                if not ids:
                    continue

                # Embed and upsert strictly in batches to limit peak RAM usage
                for batch_start in range(0, len(ids), BATCH_SIZE):
                    batch_end = batch_start + BATCH_SIZE
                    batch_texts = texts[batch_start:batch_end]
                    batch_metadatas = metadatas[batch_start:batch_end]
                    batch_ids = ids[batch_start:batch_end]

                    # Generate embeddings only for the current batch
                    batch_embeddings = list(get_embedder().embed(batch_texts))

                    collection.upsert(
                        ids=batch_ids,
                        documents=[chunk[len("passage: "):].strip() for chunk in batch_texts],
                        metadatas=batch_metadatas,
                        embeddings=batch_embeddings,
                    )
                    written_chunks += len(batch_ids)

                updated_files += 1
                print(f"Updated: {relpath} ({len(ids)} chunks)")

    print(
        f"Sync complete. updated_files={updated_files}, skipped_files={skipped_files}, written_chunks={written_chunks}"
    )
    return {
        "updated_files": updated_files,
        "skipped_files": skipped_files,
        "written_chunks": written_chunks,
    }


if __name__ == "__main__":
    sync_knowledge_base()
