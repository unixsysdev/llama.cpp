#!/bin/bash
# lb.sh LABEL BIN_DIR MODEL [extra llama-bench args...]
# Runs llama-bench with Strix Halo defaults; appends JSONL to $OUT/llama-bench.jsonl (default: this dir), prints a table.
set -u
LABEL=$1; BIN=$2; MODEL=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE}; mkdir -p "$OUT/logs"
"$BIN/llama-bench" -m "$MODEL" -ngl 999 -fa on -lm "${LM:-mmap}" "$@" -o jsonl 2> "$OUT/logs/lb-$LABEL.err" \
  | python3 -I "$HERE/lb_fmt.py" "$LABEL" "$OUT/llama-bench.jsonl"
grep -iE 'error|abort|fail' "$OUT/logs/lb-$LABEL.err" | head -5
