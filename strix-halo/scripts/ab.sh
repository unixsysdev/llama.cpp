#!/bin/bash
# ab.sh: alternate two builds on one 4096-token prompt batch, waiting for a quiet machine before each run.
#   MODEL=/path/to/first.gguf A=/path/to/old/bin B=/path/to/new/bin DEPTH=0 PAIRS=2 strix-halo/scripts/ab.sh
set -u
: "${MODEL:?MODEL=path to the first GGUF shard}" "${A:?A=bin dir of the baseline}" "${B:?B=bin dir of the candidate}"
DEPTH=${DEPTH:-0}; PAIRS=${PAIRS:-2}; QUIET_LOAD=${QUIET_LOAD:-6}

run() {
  while awk -v l="$(cut -d' ' -f1 /proc/loadavg)" -v q="$QUIET_LOAD" 'BEGIN { exit !(l >= q) }'; do sleep 10; done
  local ts
  ts=$("$2/llama-bench" -m "$MODEL" -ngl 999 -fa on -lm mmap -p 4096 -n 0 -d "$DEPTH" -b 4096 -ub 4096 -r 2 -o jsonl 2>/dev/null |
       grep -oE '"avg_ts":\s*[0-9.]+' | grep -oE '[0-9.]+$')
  echo "$1 depth=$DEPTH tok/s=$ts load=$(cut -d' ' -f1 /proc/loadavg)"
}

for _ in $(seq "$PAIRS"); do
  run A "$A"
  run B "$B"
done
