#!/bin/bash
# qwen38.sh - Qwen3.8-Flash-Next (125B MoE, ~6B active, + 51B n-gram "engram" table) server on Strix Halo
# Refs: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF   https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF
#
# Model: Unsloth UD-IQ4_XS (93.7 GB: ~65 GB GPU weights + 28.8 GB engram that stays on the CPU, memory-mapped)
#        + ggml-org MTP head (mtp-...-Q8_0, 4.1 GB) for speculative decoding.
# Needs the raised GPU memory limit (kernel cmdline: amdgpu.gttsize=126976 ttm.pages_limit=32505856).
# Binary: this branch (sparse prefill attention + coopmat lightning indexer), see README.md at the top of the repo and ../REPORT.md.
#
# Measured on this box (Vulkan/RADV, thinking on, T=1.0):
#   decode: ~27 tok/s plain, ~35 with MTP (xhigh effort), ~43 at low effort, ~100 when re-emitting code (n-gram)
#   prompt: ~810 tok/s on short prompts, ~660 at 128K deep (4096-token batches)
#   follow-up turns reuse the prompt cache: +2K tokens at 250K depth took 5 s
#
# Usage:
#   MODEL_DIR=~/models/qwen3.8-flash-next ./qwen38.sh   # 256k ctx, MTP + n-gram drafting, effort xhigh (template default)
#   EFFORT=low ./qwen38.sh       # default reasoning effort: xhigh|medium|low ("high" = xhigh)
#   THINK=0 ./qwen38.sh          # no thinking at all
#   DRAFT=0 ./qwen38.sh          # no speculative decoding
#   NGRAM=0 ./qwen38.sh          # MTP only: ~5% faster on text that never repeats, much slower when rewriting files
#   CTX=131072 ./qwen38.sh       # smaller context (less memory)
#   PARALLEL=2 ./qwen38.sh       # two slots sharing one KV pool
#   PORT=8081 HOST=127.0.0.1     # listen address
#
# Notes:
# - Keep -lm mmap: the 28.8 GB engram is read through the page cache. Any other load mode copies it into RAM,
#   and on ROCm/HIP mmap loading takes 10+ minutes (Vulkan is faster here anyway).
# - Per request: "chat_template_kwargs": {"reasoning_effort": "low"} or {"enable_thinking": false}.
# - Thinking from earlier turns is kept in the prompt (preserve_thinking), which keeps the prompt cache valid.

set -eu
HERE=$(cd "$(dirname "$0")" && pwd)

BIN="${BIN:-$HERE/../../build-vulkan/bin}"
DIR="${MODEL_DIR:-$HOME/models/qwen3.8-flash-next}"
MODEL="$DIR/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf"
MTP="$DIR/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
CTX="${CTX:-262144}"
DRAFT="${DRAFT:-2}"
NGRAM="${NGRAM:-1}"
PARALLEL="${PARALLEL:-1}"
EFFORT="${EFFORT:-}"
THINK="${THINK:-1}"
UBATCH="${UBATCH:-4096}"
BATCH="${BATCH:-4096}"
PORT="${PORT:-8081}"
HOST="${HOST:-127.0.0.1}"

[ -f "$MODEL" ] || { echo "Missing $MODEL" >&2; exit 1; }

SPEC_FLAGS=(); SPEC_LABEL="off"
if [ "$DRAFT" != "0" ]; then
  [ -f "$MTP" ] || { echo "Missing $MTP" >&2; exit 1; }
  # MTP drafts DRAFT tokens; ngram-mod takes over when the model re-emits text already in the context.
  # Probabilistic draft sampling (rejection sampling) suits T=1.0: 50-64% acceptance vs 37-45% greedy.
  SPEC_TYPE=draft-mtp,ngram-mod; SPEC_LABEL="MTP $DRAFT + ngram-mod"
  [ "$NGRAM" = "0" ] && { SPEC_TYPE=draft-mtp; SPEC_LABEL="MTP $DRAFT"; }
  SPEC_FLAGS=(-md "$MTP" --spec-type "$SPEC_TYPE" --spec-draft-n-max "$DRAFT" --spec-draft-sampling probabilistic)
fi

SLOT_FLAGS=()
[ "$PARALLEL" -gt 1 ] && SLOT_FLAGS=(--kv-unified)

KWARGS="{}"
if [ "$THINK" = "0" ]; then
  KWARGS='{"enable_thinking": false}'
elif [ -n "$EFFORT" ]; then
  KWARGS="{\"reasoning_effort\": \"$EFFORT\"}"
fi

echo "======================================================"
echo "  Qwen3.8-Flash-Next   UD-IQ4_XS"
echo "  context  : $CTX   slots: $PARALLEL   batch: $BATCH/$UBATCH"
echo "  spec     : $SPEC_LABEL"
echo "  thinking : $([ "$THINK" = 0 ] && echo off || echo "on, effort ${EFFORT:-xhigh (default)}")"
echo "  binary   : $BIN"
echo "  listen   : http://$HOST:$PORT"
echo "======================================================"

exec "$BIN/llama-server" \
  -m "$MODEL" \
  --alias "qwen3.8-flash-next" \
  -c "$CTX" \
  -np "$PARALLEL" \
  "${SLOT_FLAGS[@]}" \
  -ngl 999 \
  -fa on \
  -lm mmap \
  -b "$BATCH" -ub "$UBATCH" \
  "${SPEC_FLAGS[@]}" \
  --chat-template-kwargs "$KWARGS" \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --jinja \
  --host "$HOST" --port "$PORT"
