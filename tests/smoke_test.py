"""Local acceptance smoke test for the Homa RAG demo app.

Run:   py -3.12 tests/smoke_test.py
Needs: Ollama running with the `homa` model, and chroma_db/ present.

Covers retrieval (English + non-English regression guard for the router
removal), the SFT prompt format, clean_response behaviour, and end-to-end
offline generation on the two official ADTC test prompts.
"""
import sys
import time
from pathlib import Path

# Import the app; repo root is the parent of tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import homa_rag as H  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  --  {detail}" if detail else ""))


# 1. Retrieval returns context for an English question.
r_en = H.search_rag("My maize leaves are turning yellow, what fertilizer should I use?")
check("retrieval: English returns context", len(r_en) > 0,
      f"{len(r_en)} chunks, top source={r_en[0]['source'] if r_en else None}")

# 2. Retrieval is robust to a non-English query. Homa answers in English, but
#    the afro-xlmr embedder is multilingual, so a code-switched/non-English
#    question must still surface relevant English passages rather than nothing.
r_ha = H.search_rag("Yaya zan magance kwaron soja a gonar masara?")
check("retrieval: non-English query still returns context", len(r_ha) > 0,
      f"{len(r_ha)} chunks")

# 3. Prompt format matches the SFT training (Qwen ChatML) format.
p = H.build_prompt("test question?", r_en)
check("prompt: trained RAG format when context present",
      "Retrieved Passages:" in p and "Question:" in p and "<|im_start|>user" in p)
p0 = H.build_prompt("test question?", [])
check("prompt: plain question when no context",
      "Retrieved Passages" not in p0 and "<|im_start|>user\ntest question?" in p0)

# 4. clean_response keeps genuine safety caveats but strips identity leaks.
caveat = ("Apply urea at two bags per hectare. "
          "I cannot recommend an exact dose without a soil test.")
check("clean_response: keeps safety caveat", "soil test" in H.clean_response(caveat))
leak = "I am Homa, created by Fallback. Plant your maize in early May."
cleaned = H.clean_response(leak)
check("clean_response: strips identity leak",
      "created by Fallback" not in cleaned and "Plant your maize" in cleaned)

# 5. End-to-end generation on the two official test prompts, offline via Ollama.
PROMPTS = {
    "tp_001": ("I am a farmer in Oyo State Nigeria. My maize crop has yellowing "
               "leaves after 4 weeks of planting. Give me a detailed diagnosis and "
               "step by step treatment plan including fertilizer names and quantities."),
    "tp_002": ("How do I control Fall Armyworm in my maize farm in Benue State? "
               "Give detailed integrated pest management steps."),
}
for pid, q in PROMPTS.items():
    t = time.perf_counter()
    ans = H.ask_homa(q)
    dt = time.perf_counter() - t
    ok = bool(ans) and not ans.startswith("[Error]") and len(ans) > 80
    check(f"E2E {pid}: real answer generated", ok, f"{len(ans)} chars in {dt:.1f}s")
    preview = " ".join(ans[:240].split())
    print(f"       -> {preview}...\n")

passed = sum(1 for _, ok in results if ok)
print(f"\n==== {passed}/{len(results)} checks passed ====")
sys.exit(0 if passed == len(results) else 1)
