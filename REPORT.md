# Technical Report — Homa: An Offline Agricultural Assistant for African Farmers

**Team ID:** fallbackai-2026
**Domain:** agriculture
**Model:** Homa-Afrique-Gemma-4B (GGUF Q4_K_M)
**Submitter:** Somtochukwu Ikewelugo · sikewelugo@gmail.com · [@somto-ikewelugo](https://github.com/somto-ikewelugo)
**Weights:** [huggingface.co/fallback-ai/Homa-Afrique-Gemma-4B](https://huggingface.co/fallback-ai/Homa-Afrique-Gemma-4B)

---

## Problem

Smallholder farmers produce the majority of Africa's food, yet frontline agronomic
advice is scarce where it is needed most. Extension-officer coverage is thin, the
best crop-protection and planting guidance sits in PDFs and manuals written in
technical English, and the moment a farmer most needs help — a diseased maize leaf,
an armyworm outbreak, a fertilizer decision at week four — is exactly the moment
they are standing in a field with no reliable connectivity.

**Homa** is an offline agricultural assistant that puts practical, locally-relevant
guidance in the farmer's own language onto an affordable laptop, with no internet
dependency. It covers crop production, livestock, pest and disease management,
fertilizer decisions, and seasonal planting guidance for the Nigerian/West-African
context, and it answers in **English, Hausa, Igbo, and Yoruba**.

The target user is a rural extension worker, agro-dealer, cooperative officer, or
literate farmer operating on a budget laptop with intermittent or no connectivity.
Running fully offline is not a nice-to-have for this user — it is the whole point.
An assistant that needs the cloud is an assistant that is unavailable precisely when
and where the crop decision is being made.

---

## Design Decisions

### Base model

- **Base:** [`McGill-NLP/AfriqueGemma-4B`](https://huggingface.co/McGill-NLP) — a
  continued-pre-training of `google/gemma-3-4b-pt` over ~25.2B tokens across 20
  African languages. We chose an African-language-adapted base over a general
  instruction model so that Hausa, Igbo, and Yoruba are _native_ to the weights
  rather than bolted on, directly supporting the localisation goal.
- **Why 4B:** at Q4_K_M a 4B Gemma quantizes to ~2.5 GB on disk and runs in
  **under 5 GB RAM**, leaving comfortable headroom under the 7 GB usable ceiling.
  Smaller models (≤1.5B) lost too much agronomic reasoning and multilingual
  fidelity; 7B-class models risked the RAM ceiling once context and KV cache were
  accounted for, and were slower on integrated-GPU/CPU-only inference.

### Fine-tuning

- **Method:** LoRA (r=32, α=64) on the attention projections (q/k/v/o) and MLP
  layers, then **merged into the base weights** so the deployed artifact is a single
  self-contained GGUF with no adapter loading at runtime.
- **Framework:** TRL `SFTTrainer`, cosine LR schedule (peak 2e-4), early stopping.
- **Data:** ~5,500+ instruction–response pairs spanning crop production, livestock,
  identity/scope, out-of-domain boundary handling, and safety examples. Pairs were
  machine-translated and **hand-reviewed** across English, Hausa, Igbo, and Yoruba.
  The dataset explicitly teaches the model to _decline_ out-of-scope requests
  (market/pricing data, translation, general chit-chat) so it stays a trustworthy
  agronomy tool rather than a general chatbot.
- **RAG-aware training:** the SFT set includes the retrieval prompt format
  (`Background information … Passage N … Farmer question`) so the model uses injected
  context correctly when a retrieval layer is present, and answers directly from its
  own weights when it is not.

### Quantization

- **GGUF Q4_K_M**, built with `llama.cpp` (build b10107). Q4_K_M was chosen as the
  quality/footprint balance point: Q5/Q6 pushed peak RAM toward the ceiling for
  little qualitative gain on our test prompts, while Q3/Q2 visibly degraded
  numerical and multilingual accuracy (dosages, quantities, language consistency).

### Training result (best checkpoint, step 620)

| Metric               | Value                                         |
| -------------------- | --------------------------------------------- |
| Epochs               | 1.09 (early-stopped, no overfitting observed) |
| Eval loss            | 1.1374                                        |
| Token-level accuracy | 71.5%                                         |

---

## Constraints

- **RAM is the hard constraint.** The ADTC standard laptop allows 7 GB usable; a run
  that exceeds it is disqualified. Every quantization and context-length choice was
  made against this ceiling first, quality second.
- **CPU / integrated-GPU only.** No discrete GPU is assumed. Inference runs through
  `llama.cpp` on CPU, so tokens/second and time-to-first-token are bounded by memory
  bandwidth, not compute — reinforcing the choice of a 4B model at 4-bit.
- **100% offline.** No external network calls occur during inference. Weights are
  fetched once, ahead of evaluation, via `download_model.sh`; after that the model is
  fully self-contained.
- **Data availability.** Authoritative, region-specific agronomic sources
  (Nigerian wet-season performance surveys, IITA/CIMMYT crop manuals, Fall Armyworm
  and locust IPM guides) exist mainly as English PDFs. Making that knowledge usable
  in local languages was a core data-engineering task, not an afterthought.

---

## Benchmarks

Self-reported development benchmarks, measured with the ADTC profiler in participant
mode. Official scores are measured by the ADTC profiler on the standard evaluation
machine.

| Metric                | Value                                                    |
| --------------------- | -------------------------------------------------------- |
| Machine               | Intel Core i5 (Family 6, Model 154), 4-bit CPU inference |
| Runtime               | `llama.cpp` (GGUF Q4_K_M), architecture `gemma3`         |
| Peak RSS              | **5,570 MB (~5.4 GB)** — within the 7 GB ceiling         |
| Steady-state RSS      | 4,969 MB                                                 |
| Generation speed      | **13.79 tokens/s**                                       |
| Time to first token   | 2,113 ms                                                 |
| CPU utilisation (p99) | 55.8%                                                    |
| Thermal throttling    | **None observed**                                        |
| Native context length | 131,072 tokens (operated at 4,096)                       |

**African language support:** English, Hausa, Igbo, Yoruba — the model responds in
the language the question was asked in.

---

## Beyond the model: the Homa RAG demo

Alongside the submitted weights, FallbackAI ships an **offline agentic-RAG
application** (`homa_rag.py`) that grounds Homa's answers in a curated knowledge base
of ~950 chunks from Nigerian and pan-African agronomic sources (planting calendars,
disease and pest guides, fertilizer manuals).

The retrieval pipeline is built around a custom local embedder:

- the tokenizer is exported from `Davlan/afro-xlmr-mini` into `./afro_mini_onnx`.
- the embedding model is then quantized to INT8 and packaged as
  `./afro_mini_onnx_int8/model_quantized.onnx`.
- embeddings are computed using ONNX Runtime with `CPUExecutionProvider`, mean pooled
  over the attention mask, and L2-normalized into a 384-dimensional vector.
- text preprocessing cleans `passage:` and `query:` prefixes, then encodes the data
  in batches for efficient offline indexing.

These vectors are stored in a ChromaDB collection named `homa_knowledge_base`, and
semantic search returns the closest passages by embedding distance for injection into
the model's RAG prompt format. This design preserves the offline guarantee while
using a compact, CPU-friendly multilingual embedding stack.

This retrieval layer is the intended _product_ experience. Note that the ADTC
profiler evaluates the raw GGUF through `llama.cpp` directly, so the reported
accuracy/throughput/memory figures reflect the model on its own — the RAG layer is
demonstrated in the accompanying video rather than measured by the profiler.
