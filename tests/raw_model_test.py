"""Raw-model test — NO RAG, matching evaluation conditions.

The ADTC profiler / judges run the raw GGUF through llama.cpp with the plain
prompt (no retrieval). This sends each question straight to the `homa` model in
the trained turn format and prints the FULL, uncleaned output — i.e. exactly
what a judge sees. clean_response is shown only as a secondary reference.

Run: py -3.12 tests/raw_model_test.py   (needs Ollama serving `homa`)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import homa_rag as H  # noqa: E402

PROMPTS = [
    ("tp_001 (EN, official)",
     "I am a farmer in Oyo State Nigeria. My maize crop has yellowing leaves "
     "after 4 weeks of planting. Give me a detailed diagnosis and step by step "
     "treatment plan including fertilizer names and quantities."),
    ("tp_002 (EN, official)",
     "How do I control Fall Armyworm in my maize farm in Benue State? Give "
     "detailed integrated pest management steps."),
    ("out-of-domain (should decline)",
     "What is the current market price of a bag of maize in Lagos today?"),
]


def raw_generate(question):
    """Plain question in the trained format, straight to the model. No RAG."""
    prompt = H.build_prompt(question, [])  # empty context -> plain question turn
    import requests
    t = time.perf_counter()
    r = requests.post(
        H.OLLAMA_URL,
        json={
            "model": "homa",
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 1024,
                "num_ctx": 4096,
                "stop": ["<|im_end|>", "<|im_start|>"],
            },
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip(), time.perf_counter() - t


for label, q in PROMPTS:
    print("=" * 78)
    print(f"### {label}")
    print(f"Q: {q}")
    ans, dt = raw_generate(q)
    print(f"\n--- RAW model output ({len(ans)} chars, {dt:.1f}s) ---")
    print(ans if ans else "(empty)")
    cleaned = H.clean_response(ans)
    if cleaned != ans:
        print(f"\n--- after clean_response ({len(cleaned)} chars) ---")
        print(cleaned)
    print()
