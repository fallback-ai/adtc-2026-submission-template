
import shutil
import hashlib
import os
import re
import subprocess
from pathlib import Path

import chromadb
import pymupdf
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
    """ONNX-based embedder for multilingual `Davlan/afro-xlmr-mini`.

    This class loads a tokenizer export from `./afro_mini_onnx` and performs INT8
    inference using ONNX Runtime with a quantized model bundle.
    """

    def __init__(self):
        """Initialize tokenizer and INT8 ONNX inference session."""
        # Load the tokenizer from the base export folder
        self.tokenizer = AutoTokenizer.from_pretrained("./afro_mini_onnx")

        # Load the INT8 quantized weights
        self.session = ort.InferenceSession(
            "./afro_mini_onnx_int8/model_quantized.onnx",
            providers=["CPUExecutionProvider"],
        )

    def _clean_texts(self, texts):
        """Normalize text inputs by stripping special RAG prefixes and whitespace."""
        cleaned = []
        for text in texts:
            text = text.replace("passage: ", "").replace("query: ", "")
            cleaned.append(text.strip())
        return cleaned

    def embed(self, texts):
        """Convert input texts to normalized ONNX embedding vectors.

        Args:
            texts (list[str]): A list of text strings to embed.

        Returns:
            list[list[float]]: A list of normalized embedding vectors.
        """
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
        mask_sum = np.clip(mask.sum(axis=1, keepdims=True),
                           a_min=1e-9, a_max=None)
        pooled = (last_hidden_state * mask[:, :, None]).sum(axis=1) / mask_sum
        normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return normalized.tolist()


def get_embedder():
    """Return a singleton embedder instance, building the ONNX bundle if needed."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        target_quant_file = Path("./afro_mini_onnx_int8/model_quantized.onnx")

        if not target_quant_file.exists():
            print(
                "Model bundle not found. Building INT8 ONNX model for Davlan/afro-xlmr-mini...")
            try:
                # Remove any broken/incomplete export folders from previous attempts
                if Path("./afro_mini_onnx").exists():
                    shutil.rmtree("./afro_mini_onnx")
                if Path("./afro_mini_onnx_int8").exists():
                    shutil.rmtree("./afro_mini_onnx_int8")

                print("Step 1/2: Exporting base PyTorch model to standard ONNX...")
                subprocess.run(
                    "optimum-cli export onnx --model Davlan/afro-xlmr-mini --task feature-extraction ./afro_mini_onnx",
                    shell=True,
                    check=True,
                )

                print("Step 2/2: Quantizing ONNX model to INT8 (AVX2 optimized)...")
                subprocess.run(
                    "optimum-cli onnxruntime quantize --avx2 --onnx_model ./afro_mini_onnx -o ./afro_mini_onnx_int8",
                    shell=True,
                    check=True,
                )
            except Exception as e:
                print(f"[error] optimum-cli failed: {e}")
                raise RuntimeError(
                    "Failed to build ONNX model bundle. Cannot proceed.")

        print("Loading AfroXLMR-Mini ONNX embedder...")
        _EMBEDDER_INSTANCE = AfroXLMRMiniEmbedder()
        print("Model loaded successfully!")
    return _EMBEDDER_INSTANCE


client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = client.get_or_create_collection("homa_knowledge_base")
    # Probe check to catch vector dimension mismatch early
    if collection.count() > 0:
        sample = collection.get(limit=1, include=["embeddings"])
        if sample.get("embeddings") is not None and len(sample["embeddings"]) > 0:
            if len(sample["embeddings"][0]) != 384:
                raise ValueError("Dimension mismatch detected in ChromaDB")
except Exception as e:
    print(
        f"[info] Resetting ChromaDB collection due to model schema update: {e}")
    try:
        client.delete_collection("homa_knowledge_base")
    except Exception:
        pass
    collection = client.create_collection("homa_knowledge_base")


def clean_text(text):
    """Normalize file text to safe UTF-8 whitespace and punctuation.

    Non-ASCII control characters are removed, line endings are normalized, and
    consecutive whitespace is collapsed.
    """
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def extract_text(filepath):
    """Extract and clean text from supported file formats.

    Args:
        filepath (str or Path): Path to a PDF or text file.

    Returns:
        list[tuple[int, str]]: A list of (page_number, cleaned_text) pairs.
    """
    filepath = str(filepath)
    if filepath.lower().endswith(".pdf"):
        pages = []
        with pymupdf.open(filepath) as doc:
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
    """Split a text block into chunks while preserving sentence boundaries."""
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
    """Chunk cleaned text into size-limited segments with controlled overlap."""
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
    """Normalize a source path to forward-slash format for metadata consistency."""
    return source_path.replace("\\", "/")


def _chunk_id(relpath, page_num, chunk_idx, chunk_text):
    """Compute a stable chunk ID from source path, page, and text prefix."""
    digest = hashlib.md5(
        f"{relpath}|{page_num}|{chunk_idx}|{chunk_text[:40]}".encode("utf-8")
    )
    return digest.hexdigest()


def _load_existing_files(collection):
    """Read metadata from the existing ChromaDB collection.

    Returns a mapping of source path to stored IDs and modification times.
    """
    existing = {}
    try:
        results = collection.get(include=["metadatas", "documents"])
    except Exception:
        return existing

    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])

    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if ids and isinstance(ids[0], list):
        ids = ids[0]

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


def _source_is_complete(existing, expected_ids, mtime):
    """Return True when a source file is fully indexed and unchanged."""
    if not existing:
        return False
    if existing.get("mtime") != mtime:
        return False
    stored_ids = set(existing.get("ids", []))
    return stored_ids == set(expected_ids)


def _delete_source_chunks(collection, ids_to_delete):
    """Delete existing chunk IDs for a source, ignoring deletion failures."""
    if not ids_to_delete:
        return
    try:
        collection.delete(ids=ids_to_delete)
    except Exception as e:
        print(f"[warn] failed to delete partial source chunks: {e}")


def _build_file_chunks(relpath, category, mtime, pages):
    """Build chunk IDs, texts, and metadatas for a file's extracted pages."""
    ids = []
    texts = []
    metadatas = []
    for page_num, page_text in pages:
        page_chunks = chunk_text(page_text)
        for chunk_idx, chunk in enumerate(page_chunks):
            if len(chunk.strip()) < 50:
                continue
            chunk_id = _chunk_id(relpath, page_num, chunk_idx, chunk)
            ids.append(chunk_id)
            texts.append(f"passage: {chunk}")
            metadatas.append({
                "source": relpath,
                "category": category,
                "page": page_num,
                "file_mtime": mtime,
            })
    return ids, texts, metadatas


def sync_knowledge_base():
    """Index local knowledge base documents into ChromaDB.

    This function walks the knowledge base folders, extracts and chunks text,
    computes embeddings, and updates the ChromaDB collection while preserving
    existing entries for unchanged files.
    """
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

                pages = extract_text(filepath)
                if not pages:
                    continue

                ids, texts, metadatas = _build_file_chunks(
                    relpath, category, mtime, pages
                )
                if not ids:
                    continue

                if _source_is_complete(existing, ids, mtime):
                    print(f"Skipped: {relpath} (already complete)")
                    skipped_files += 1
                    continue

                if existing and existing.get("ids"):
                    print(
                        f"Dirty: {relpath} (partial or stale index) - removing {len(existing['ids'])} old chunks")
                    _delete_source_chunks(collection, existing["ids"])
                else:
                    print(f"Processing: {relpath}")

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
                        documents=[chunk[len("passage: "):].strip()
                                   for chunk in batch_texts],
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
