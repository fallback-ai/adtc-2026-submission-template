# Testing Homa — setup & run guide

Everything a teammate needs to set up and run **all** the tests after pulling
the `development` branch. Commands are PowerShell (Windows). Run them from the
repo root unless noted.

> **Two things trip everyone up, read first:**
> 1. Use **Python 3.12** (`py -3.12`) for every Python command. The default
>    `python` on these machines is 3.14 and has none of the deps installed.
> 2. Ollama **snapshots the GGUF when you run `ollama create`.** If the model
>    file changes later, the old tag keeps the old weights — you must
>    `ollama create` again. Dropping a new file in `model/` does nothing on its own.

---

## 0. One-time prerequisites

Install these once:

- **Python 3.12** — `py -3.12 --version` should work.
- **Ollama** — https://ollama.com/download. Make sure the server is running
  (the desktop app starts it; or run `ollama serve` in a spare terminal).
- **llama.cpp `llama-bench.exe`** — needed by the profiler's throughput test.
  Download a llama.cpp release build and put the folder on PATH for the session:
  ```powershell
  $env:PATH += ";C:\path\to\llama.cpp\bin"   # folder containing llama-bench.exe
  llama-bench --version                        # verify
  ```
- **ADTC reference profiler** — install per the ADTC instructions so that
  `py -3.12 -c "import adtc_profiler"` succeeds. (Used for throughput/RAM and,
  optionally, accuracy.)
- **lm-eval** (only if you want to run the accuracy path locally):
  ```powershell
  py -3.12 -m pip install "lm-eval[gguf]"
  ```

Confirm Ollama is up:
```powershell
(Invoke-RestMethod http://localhost:11434/api/tags).models.name
```

---

## 1. Get the model weights and build the Ollama tag

The GGUF is **not** in git (it's ~0.98 GB). Download it from Hugging Face, then
build the `homa` Ollama tag from the Modelfile.

```powershell
# Downloads model/homa-qwen15b-q4.gguf from HF (public, idempotent).
# Needs Git Bash / WSL for the .sh, OR use the curl line below directly.
bash download_model.sh
```

If you don't have bash, download directly:
```powershell
curl.exe -L --fail -o "model\homa-qwen15b-q4.gguf" `
  "https://huggingface.co/fallback-ai/Homa-Qwen2.5-1.5B/resolve/main/homa-qwen15b-q4.gguf"
```

Build the Ollama model (the test scripts all use the tag **`homa`**):
```powershell
ollama create homa -f Modelfile
ollama run homa "Who are you?"      # quick sanity check; /bye or Ctrl-D to exit
```

---

## 2. Build the RAG vector store (for the demo + smoke test)

`chroma_db/` ships empty. Populate it from `knowledge_base/`:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 embedder.py                 # runs sync_knowledge_base() -> fills chroma_db/
```

This ingests/embeds the knowledge-base docs. Re-running is incremental (only
changed files re-embed). You only need this for the RAG demo and `smoke_test.py`
— the raw-model and profiler tests below do **not** need it.

---

## 3. Run the tests

### 3a. Interactive sanity check (with Modelfile: template + identity SYSTEM)
```powershell
ollama run homa
# try:  Hi   /   Who are you?   /   Are you ChatGPT?   /
#       What are the best crops to farm in Jigawa, Nigeria?   /
#       (follow-up) Which of those does best in poor sandy soil?   <- multi-turn
```

### 3b. RAG end-to-end smoke test (needs chroma_db/ + Ollama `homa`)
Retrieval, prompt format, `clean_response`, and offline generation
on the two official ADTC prompts. Exits non-zero if any check fails.
```powershell
py -3.12 tests/smoke_test.py
```

### 3c. Raw-model test — NO RAG, judge-like conditions (needs Ollama `homa`)
Sends each prompt straight to the raw model in the trained turn format via
`raw:true` and prints the full uncleaned output — closest to what a judge sees.
```powershell
py -3.12 tests/raw_model_test.py
```
To test the **pure bare weights with no system preamble at all** (exactly the
audited model, no runtime nudge), disable the injected system first:
```powershell
$env:HOMA_SYSTEM_PROMPT = ""
py -3.12 tests/raw_model_test.py
Remove-Item Env:\HOMA_SYSTEM_PROMPT     # restore afterwards
```

### 3d. The profiler — throughput + RAM (this is the SCORED perf path)
Runs `llama-bench` on the raw GGUF and writes a schema-valid `submission.json`.
**Do this on a quiet machine** (close other heavy apps, and unload the model from
Ollama first) or the throughput reading is contaminated.
```powershell
# free the machine: unload any resident Ollama model
Invoke-RestMethod -Uri http://localhost:11434/api/chat -Method Post `
  -Body '{"model":"homa","messages":[],"keep_alive":0}' -ContentType "application/json" | Out-Null

$env:PYTHONUTF8 = "1"
adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy
```
If the `adtc-profiler` executable is blocked (Application Control) or not on
PATH, invoke the module instead:
```powershell
$env:PYTHONUTF8 = "1"
py -3.12 -c "from adtc_profiler.cli import main; main()" run `
  --submission . --mode participant --output submission.json --skip-accuracy
```
Read the result:
```powershell
$s = Get-Content submission.json | ConvertFrom-Json
$s.throughput; $s.memory; $s.cpu_thermal
```

### 3e. The profiler — accuracy (optional; the judges' accuracy harness)
Same command **without** `--skip-accuracy`. It shells out to
`lm_eval --model gguf --pretrained <model>`. Needs `lm-eval` installed and
`llama-bench`/llama on PATH.
```powershell
$env:PYTHONUTF8 = "1"
adtc-profiler run --submission . --mode participant --output submission.json `
  --accuracy-task arc_easy --accuracy-limit 50
```
> Note: `arc_easy` is only a plumbing sanity check. The real audit scores a
> **hidden agriculture validation subset**, so treat this number as "does the
> harness run", not as your competition accuracy.

### 3f. Optional: run the RAG demo by hand
```powershell
py -3.12 homa_rag.py
# type farming questions; 'reset' clears history, 'quit' exits
```

---

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` (requests/chromadb/fastembed) | You used `python` (3.14). Use `py -3.12`. |
| Model answers as an old version / OpenAI identity | Old Ollama tag. Re-run `ollama create homa -f Modelfile`. |
| `[Error] Couldn't reach the model` | Ollama server not running. Start the app or `ollama serve`. |
| Unicode/garbled console output | `Set` `$env:PYTHONUTF8 = "1"` before the command. |
| Profiler throughput looks low/unstable | Machine was busy. Unload Ollama (see 3d) and close other apps; re-run. |
| `llama-bench not found` | Add its folder to PATH for the session (see prereqs). |
| `smoke_test.py` retrieval FAIL | `chroma_db/` not built. Run `py -3.12 embedder.py`. |
| `adtc-profiler` blocked / not found | Use the `py -3.12 -c "from adtc_profiler.cli import main; main()"` form. |

## What maps to the judges

- **Scored automated path = raw GGUF only.** Throughput (`llama-bench`) and
  accuracy (`lm_eval` on the `.gguf`). The Ollama **Modelfile is NOT used here** —
  identity/behaviour must live in the weights (they do, verified). Reproduce with
  3d/3e and the bare-weights variant of 3c.
- **Human/demo path = Modelfile applies.** `ollama run homa` and `homa_rag.py`
  use the multi-turn template + identity SYSTEM. Reproduce with 3a/3b.
