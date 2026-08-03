import requests
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

ROUTING_KEYWORDS = {
    "crop_disease": ["disease", "mosaic", "blight", "wilt", "rot", "spots", "lesion", "infection", "fungus", "virus", "streak"],
    "fertilizer": ["fertilizer", "yellow", "nutrient", "nitrogen", "NPK", "urea", "deficiency", "manure", "chlorosis", "pale"],
    "planting": ["plant", "when", "season", "harvest", "variety", "seed", "spacing", "calendar", "maize", "cassava", "yam", "rice", "tomato", "grow"],
    "pest_management": ["pest", "insect", "worm", "armyworm", "locust", "caterpillar", "spray", "pesticide", "control", "damage"]
}

def route_query(question):
    q = question.lower()
    matched = []
    for collection, keywords in ROUTING_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            matched.append((collection, score))
    matched.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in matched[:2]]

def search_rag(question, n_results=2):
    embedding = embedder.encode(question).tolist()
    collections_to_search = route_query(question)
    results_list = []
    if collections_to_search:
        for col_name in collections_to_search:
            try:
                results = COLLECTIONS[col_name].query(
                    query_embeddings=[embedding],
                    n_results=n_results
                )
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    results_list.append({"text": doc, "source": (meta or {}).get("source", "unknown")})
            except Exception as e:
                print(f"[warn] search failed for collection '{col_name}': {e}")
    return results_list[:3]

def build_prompt(question, context_docs):
    if context_docs:
        passages = "\n\n".join([
            f"Passage {i+1}:\n{r['text']}"
            for i, r in enumerate(context_docs)
        ])
        user_content = (
            f"Background information:\n\n{passages}\n\n"
            f"Farmer question: {question}\n\n"
            f"You are Homa, a farming helper. Answer like you are talking to a local farmer "
            f"who has no scientific training. Use simple, everyday words. "
            f"You may name the disease or pest plainly, but avoid Latin names and jargon. "
            f"Give clear, practical steps the farmer can act on today. "
            f"Always reply in the same language the farmer used to ask the question. "
            f"Do not apologise or explain your limitations. "
            f"Never mention documents, passages or reference material. "
            f"If the background information is not relevant, answer from your own knowledge."
        )
    else:
        user_content = (
            f"Farmer question: {question}\n\n"
            f"You are Homa, a farming helper. Answer like you are talking to a local farmer "
            f"who has no scientific training. Use simple, everyday words. "
            f"Give clear, practical steps the farmer can act on today. "
            f"Always reply in the same language the farmer used to ask the question. "
            f"Do not apologise or explain your limitations."
        )
    return f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"

def clean_response(answer):
    # Only drop sentences that leak the model's plumbing: its identity, its
    # instructions, or the RAG passages. Do NOT strip genuine safety caveats or
    # expressions of uncertainty ("consult the label", "I cannot recommend an
    # exact dose without a soil test", "see an extension officer") — for advice
    # on pesticide/fertilizer dosages those warnings must reach the farmer.
    # Deflection ("just go see an extension officer" instead of answering) is a
    # prompting/fine-tuning problem, not something to paper over by deletion.
    leak_phrases = [
        "i am homa",
        "my instructions",
        "scientific jargon",
        "created by fallback",
        "please upload", "upload them",
        "the provided text",
        "does not cover",
    ]
    sentences = answer.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|")
    clean = [s for s in sentences if not any(p in s.lower() for p in leak_phrases)]
    return " ".join(clean).strip()

def ask_homa(question):
    context_docs = search_rag(question)
    prompt = build_prompt(question, context_docs)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "homa",
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 1024,
                "num_ctx": 4096,
                "stop": ["<start_of_turn>", "<end_of_turn>"]
            }
        }
    )
    answer = response.json().get("response", "").strip()
    return clean_response(answer)

print("Homa RAG Agent ready!")
print("Type your farming question. Type quit to exit.\n")
while True:
    question = input("You: ").strip()
    if question.lower() in ["quit", "exit"]:
        break
    if not question:
        continue
    answer = ask_homa(question)
    print(f"\nHoma: {answer}\n")
