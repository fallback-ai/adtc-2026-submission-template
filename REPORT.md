# Technical Report — Homa: An Offline Agricultural Assistant for African Farmers

**Team ID:** fallbackai-2026
**Domain:** agriculture
**Model:** Homa-Qwen2.5-1.5B (GGUF Q4_K_M)
**Submitter:** Somtochukwu Ikewelugo · sikewelugo@gmail.com · [@somto-ikewelugo](https://github.com/somto-ikewelugo)
**Weights:** [huggingface.co/fallback-ai/Homa-Qwen2.5-1.5B](https://huggingface.co/fallback-ai/Homa-Qwen2.5-1.5B)

---

## Problem

Smallholder farmers produce the majority of Africa's food, yet frontline agronomic
advice is scarce where it is needed most. Extension-officer coverage is thin, the
best crop-protection and planting guidance sits in PDFs and manuals written in
technical English, and the moment a farmer most needs help — a diseased maize leaf,
an armyworm outbreak, a fertilizer decision at week four — is exactly the moment
they are standing in a field with no reliable connectivity.

**Homa** is an offline agricultural assistant that puts practical, locally-relevant
guidance onto an affordable laptop, with no internet dependency. It covers crop
production, livestock, pest and disease management, fertilizer decisions, and
seasonal planting guidance for the Nigerian/West-African context.

The target user is a rural extension worker, agro-dealer, cooperative officer, or
literate farmer operating on a budget laptop with intermittent or no connectivity.
Running fully offline is not a nice-to-have for this user — it is the whole point.
An assistant that needs the cloud is an assistant that is unavailable precisely when
and where the crop decision is being made.

---

## Design Decisions

### Base model

- **Base:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  — a strong, compact instruction model with reliable numeric reasoning (dosages,
  seed rates, bag/hectare arithmetic) and a native ChatML system role, which lets us
  place Homa's identity cleanly in a system turn rather than folding it into the
  prompt body.
- **Why 1.5B:** the ADTC target device is a 4 vCPU / 8 GB laptop with integrated
  graphics running `llama.cpp` on CPU. At Q4_K_M a 1.5B model quantizes to **~0.98 GB
  on disk** and runs at **~1.1 GB peak RAM**, which leaves almost the entire 7 GB
  usable budget free and roughly triples generation throughput versus the 4B class on
  the CPU-only, portable (no-SIMD) build the audit uses.

### Language scope

Homa operates in **English**. We prototyped an African-language-adapted 4B base
(Gemma) and a bilingual/quadrilingual corpus, but in the sub-2B class needed to hit
the throughput and RAM targets on the audit device, non-English generation was not
robust: models without African languages in their pretraining collapsed into
repetition or wrong-language output that could not be repaired by fine-tuning alone.
Rather than ship an unreliable multilingual claim, we scoped to English — where the
model is strong and dependable — and forgo the African Alpha language bonus while
retaining the African **use-case** focus (Nigerian crops, agro-inputs, and pest/
disease context). The retrieval embedder remains multilingual, so a code-switched or
non-English question still surfaces the right English source passages.

### Fine-tuning

- **Method:** LoRA (r=32, α=64) on the attention projections (q/k/v/o) and MLP
  layers, then **merged into the base weights** so the deployed artifact is a single
  self-contained GGUF with no adapter loading at runtime.
- **Framework:** TRL `SFTTrainer`, ChatML turn format with a fixed Homa system
  message, completion-only loss on the assistant turn, cosine LR schedule, 3 epochs.
- **Data:** ~1,900 English instruction–response pairs spanning crop production,
  livestock, pest and disease diagnosis, fertilizer decisions, identity/scope, and
  out-of-domain boundary handling. The set explicitly teaches the model to _decline_
  out-of-scope requests (live market/pricing data, translation, general chit-chat)
  so it stays a trustworthy agronomy tool rather than a general chatbot.
- **RAG-aware training:** the SFT set includes the retrieval prompt format
  (`Retrieved Passages … Passage N … Question`) so the model uses injected context
  correctly when a retrieval layer is present, and answers directly from its own
  weights when it is not.

### Quantization

- **GGUF Q4_K_M**, built with `llama.cpp`. Q4_K_M was chosen as the quality/footprint
  balance point: Q5/Q6 add RAM for little qualitative gain on our test prompts, while
  Q3/Q2 visibly degraded numerical accuracy (dosages, quantities, bag sizes) — which
  matters more on a 1.5B model than a larger one. K-quants (not i-quants) are used
  deliberately: i-quants are GPU-oriented and dequantize more slowly on the CPU-only
  audit device.

---

## Constraints

- **RAM is a hard constraint.** The ADTC standard laptop allows ~7 GB usable; a run
  that exceeds it is disqualified. Homa's ~1.1 GB peak leaves a very large margin,
  which also protects against OOM once context and KV cache grow.
- **CPU / integrated-GPU only, portable build.** No discrete GPU is assumed. The
  audit builds `llama.cpp` with `GGML_NATIVE=OFF` and all SIMD (AVX/AVX2/FMA) **off**,
  so inference runs on scalar CPU kernels. Throughput and time-to-first-token are
  bounded by this portable build, not by an AVX2 binary — reinforcing the choice of a
  small model at 4-bit. All development numbers below are measured on this same
  portable build for comparability.
- **100% offline.** No external network calls occur during inference. Weights are
  fetched once, ahead of evaluation, via `download_model.sh`; after that the model is
  fully self-contained.
- **Data availability.** Authoritative, region-specific agronomic sources
  (Nigerian wet-season performance surveys, IITA/CIMMYT crop manuals, Fall Armyworm
  and locust IPM guides) exist mainly as English PDFs; these back both the fine-tuning
  data and the retrieval knowledge base.

---

## Benchmarks

Self-reported development benchmarks, measured with the ADTC profiler in **participant
mode** inside the profiler's own Docker image (portable no-SIMD `llama.cpp`, pinned to
a 4-core / 7.5 GB envelope) to mirror the audit environment. Official scores are
measured by the ADTC profiler on the standard evaluation machine.

| Metric                | Value                                                        |
| --------------------- | ------------------------------------------------------------ |
| Machine               | Intel i7-1270P, CPU-only, 4-core envelope (Docker)           |
| Runtime               | `llama.cpp` (GGUF Q4_K_M), architecture `qwen2`, portable build |
| Parameters            | 1.54B (matches "1.5B" claim)                                 |
| Peak RSS              | **~1.1 GB** — far under the 7 GB ceiling                     |
| Steady-state RSS      | ~1.0 GB                                                      |
| Generation speed      | **~6 tokens/s** (portable no-SIMD build)                    |
| Time to first token   | ~53 s (512-token prompt prefill on scalar kernels)          |
| Thermal throttling    | **None observed**                                            |
| Native context length | 32,768 tokens (operated at 4,096)                            |

**Reading the numbers.** The generation figure reflects the mandatory portable build,
not an AVX2 binary (which is ~4× faster but is *not* what the audit runs). It is
cross-checked against the real ADTC audit: the earlier 4B model scored ~2.4 tokens/s
on the audit VM, and param-scaling to 1.5B predicts ~6.3 tokens/s — consistent with
what we measure. Against the scoring formula this yields a strong efficiency score
(`S_eff ≈ 84`, from ~1.1 GB peak) and a much-improved throughput score
(`S_perf ≈ 41`, versus ~16 for the 4B).

---

## Beyond the model: the Homa RAG demo

Alongside the submitted weights, FallbackAI ships an **offline agentic-RAG
application** (`homa_rag.py`) that grounds Homa's answers in a curated knowledge base
built from Nigerian and pan-African agronomic sources (planting calendars, disease
and pest guides, fertilizer manuals).

The retrieval pipeline is built around a custom local embedder:

- the tokenizer is exported from `Davlan/afro-xlmr-mini` into `./afro_mini_onnx`.
- the embedding model is quantized to INT8 and packaged as
  `./afro_mini_onnx_int8/model_quantized.onnx`.
- embeddings are computed with ONNX Runtime (`CPUExecutionProvider`), mean-pooled
  over the attention mask, and L2-normalized into a 384-dimensional vector.
- text preprocessing cleans `passage:` and `query:` prefixes, then encodes in batches
  for efficient offline indexing.

These vectors are stored in a ChromaDB collection named `homa_knowledge_base`, and
semantic search returns the closest passages by embedding distance for injection into
the model's RAG prompt format. A multilingual embedder over an English knowledge base
means a non-English question still retrieves the right passages, even though Homa
answers in English. This design preserves the offline guarantee end to end.

This retrieval layer is the intended _product_ experience. Note that the ADTC profiler
evaluates the raw GGUF through `llama.cpp` directly, so the reported
accuracy/throughput/memory figures reflect the model on its own — the RAG layer is
demonstrated in the accompanying video rather than measured by the profiler.
