# Profiling & Scoring — how Homa is judged, and how to measure it right

Reference notes distilled from the `adtc-profiler` repo + source, and what they
mean for our Q4_K_M submission. Read this before regenerating `submission.json`.

---

## 1. The scoring formula

```
S_total = 0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal

  S_perf    = min(TPS / 15.0, 1.0) × 100        # TPS = generation tokens/sec; capped at 15 t/s
  S_eff     = max(0, (7.0 − peak_rss_GB) / 7.0) × 100   # lower RAM = more points, linear to 7 GB
  P_thermal = 10-point deduction if the CPU throttles or core temp > 85 °C
```

- **Accuracy is 50%** — scored on participant prompts + domain prompts + a
  **hidden agriculture subset** supplied by judges. This is the biggest lever and
  the only remaining upside (see §6).
- **Throughput is not capped until 15 t/s** — every t/s below that is real points.
- **Memory efficiency rewards low RAM**, linearly, up to the 7 GB usable ceiling.
- **Thermal**: cloud VMs report no temp sensor → `throttled=false` → `P_thermal = 0`.

### Where Q4_K_M lands (healthy Linux audit box)

| Component | Value | Score | Weighted |
| --- | --- | --- | --- |
| S_perf | ~10 t/s → 10/15 | 67 | ~20 / 30 |
| S_eff | ~2.6 GB (Linux mmap) → (7−2.6)/7 | 63 | ~12.6 / 20 |
| P_thermal | throttled=false, temp=null | 0 | 0 |
| **Non-accuracy banked** | | | **~32.6 / 50** |

So `S_total ≈ 0.5·S_acc + 32.6`.

---

## 2. Two different numbers: the SCORE vs the CONSISTENCY CHECK

- The **score** is computed from the **audit run** (`measured_on: audit_cloud_vm`,
  a Linux cloud VM the ADTC orchestrator runs). Our `submission.json` does **not**
  set the score.
- `submission.json` (`measured_on: participant_laptop`) is checked for
  **consistency** against the audit by the profiler's `comparator`.

### Comparator tolerances (submission vs audit)

| Field | Tolerance | Beyond 2× tolerance → |
| --- | --- | --- |
| `memory.peak_rss_mb` | ±15% | **fail** |
| `memory.steady_state_rss_mb` | ±15% | **fail** |
| `throughput.tokens_per_second_generation` | ±25% | **fail** |
| `throughput.first_token_latency_ms` | ±25% | **fail** |

Verdict ladder: **pass** (all within tolerance) → **flag** (within 2×) →
**fail** (beyond 2×, or schema-invalid / `team_id` mismatch). Accuracy is *not*
diffed (participant = public benchmark, audit = hidden subset; passed through).

---

## 3. ⚠️ Measure `submission.json` on LINUX (the trap)

`peak_rss_mb` depends heavily on OS memory accounting:

| Environment | Q4_K_M peak RSS |
| --- | --- |
| Windows (working set) | ~5,480 MB |
| Linux (mmap RSS) | ~2,600 MB |

The audit runs on **Linux**, where it reads ~2.6 GB. A Windows-measured
submission (5.48 GB) differs by ~53% → **fails** the ±15% memory checks even
though the model is fine. **Always generate the official `submission.json` on
Linux.** Keep any Windows run only as a local note.

Same logic for throughput: measure on a machine comparable to the audit
(non-burstable, ~4 vCPU) so it lands within ±25%.

---

## 4. Linux runbook

### 4.0 Pick a sane machine (matters most)
Non-burstable, ~4-vCPU, local SSD. Verify:
```bash
nproc                          # ~4 (the target profile)
cat /sys/fs/cgroup/cpu.max     # "max 100000" (uncapped), NOT a small quota
```
Burstable VMs with exhausted credits give ~2 t/s and 150 s TTFT — that's the
machine, not the model. If you see that, switch instances.

### 4.1 One-time setup
```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip
python3.12 -m pip install --user "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler"
export PATH="$PATH:$HOME/.local/bin"

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j && cd ..
export PATH="$PATH:$PWD/llama.cpp/build/bin"
llama-bench --version          # header should show AVX2 = 1
```

### 4.2 Model
```bash
cd adtc-2026-submission-template
./download_model.sh                                   # Q4_K_M -> model/
cat model/homa-qwen15b-q4.gguf > /dev/null            # pre-warm page cache
```

### 4.3 Measure (participant mode)
```bash
adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy
# if the entrypoint isn't on PATH:
python3.12 -c "from adtc_profiler.cli import main; main()" run \
  --submission . --mode participant --output submission.json --skip-accuracy
```
`--skip-accuracy` matches our `accuracy: []` convention (judges score the hidden
subset). Drop it only if you want the local arc_easy plumbing number (`lm-eval`
required).

### 4.4 Verify
```bash
python3.12 -c "import json;d=json.load(open('submission.json'));print(d['environment']['measured_on'], d['throughput'], d['memory'])"
```
Expect `participant_laptop`, **peak_rss_mb ~2600–4000**, throughput ~audit-ballpark.

### 4.5 Self-audit (prove it will pass)
```bash
adtc-profiler run --submission . --mode audit --output audit.json --skip-accuracy
adtc-profiler compare --help                 # confirm flags on your version
adtc-profiler compare --submission submission.json --audit audit.json
```
A `pass` verdict = RSS within ±15% and throughput within ±25%.

### Gotchas
- Don't pin fewer threads unless the box has >4 cores (then wrap llama-bench with
  `-t 4` to represent the 4-vCPU target).
- CPU-only: no discrete-GPU offload (`integrated GPU only` per the ADTC profile).
- Keep `num_ctx` at 4096 — a larger context grows the KV cache and RSS, costing
  S_eff points.

---

## 5. Quantization decision (settled)

The submission ships **Q4_K_M**. We tested alternatives on CPU (the target):

| Quant | Size | Gen t/s (4 threads) | Verdict |
| --- | --- | --- | --- |
| **Q4_K_M** | 2.48 GB | 10.2 | **ship** — accuracy-safe |
| IQ4_XS | 2.26 GB | 9.8 | reject — i-quants are GPU-optimized; **slower on CPU** (pp 46 vs 167 t/s) |
| Q3_K_M | 2.09 GB | 11.8 (+16%) | reject — breaks numerics (bag arithmetic "500 g", "600 kg/ha" dosage, seed-rate deflection) |

**Rule for a CPU target: use K-quants / legacy quants, never `IQ*` (i-quants).**
The quant lever for S_perf/S_eff is exhausted; Q4_K_M is the accuracy/speed
balance point.

---

## 6. Where the remaining points are

- **S_acc (50%)** is the only real upside left. Known soft spots: the
  maize-streak → cassava-mosaic misdiagnosis and occasional dosage/numeric slips.
  Improving these is a **data / V3.1 retrain** task, not a profiler or quant task.
- **S_perf** depends mostly on the audit machine's health — confirm with
  organizers that audits run on properly-provisioned (non-burstable) hardware.
- **S_eff** and **thermal** are locked in favorably; keep RAM low (Linux, ctx 4096).
