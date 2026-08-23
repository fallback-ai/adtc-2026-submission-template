#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/homa-qwen15b-q4.gguf"
MODEL_URL="https://huggingface.co/fallback-ai/Homa-Qwen2.5-1.5B/resolve/main/v1/homa-qwen15b-q4.gguf"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
  echo "model already present at $MODEL_FILE — skipping download"
  exit 0
fi

# clean up partial file if interrupted (Ctrl-C, error, etc.)
trap 'echo "interrupted — partial file kept for resume at $MODEL_FILE.partial"' INT TERM

echo "downloading $MODEL_URL → $MODEL_FILE (~990 MB)…"

if command -v curl > /dev/null 2>&1; then
  curl -L --fail --progress-bar -C - -o "$MODEL_FILE.partial" "$MODEL_URL"
elif command -v wget > /dev/null 2>&1; then
  wget --show-progress -c -O "$MODEL_FILE.partial" "$MODEL_URL"
else
  echo "error: neither curl nor wget found" >&2
  exit 1
fi

mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "download complete.
echo "model downloaded at: $MODEL_FILE"