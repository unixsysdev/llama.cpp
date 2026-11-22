#!/bin/bash
# run_server_bench.sh LABEL BIN_DIR MODEL [extra llama-server args...]
# Env: PORT (8099), PROMPTS (code,chat), REPS (2), MAXTOK (512), TEMP (1.0), TOPK (20), EFFORT, CTX (32768), DUMP (dir), LM (mmap)
# Starts llama-server, waits for /health, runs bench_server.py, stops server.
# Appends JSON lines to results.jsonl and keeps the server log in logs/.
set -u
LABEL=$1; BIN=$2; MODEL=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8099}; CTX=${CTX:-32768}
mkdir -p "$HERE/logs"
LOG="$HERE/logs/$LABEL.log"
"$BIN/llama-server" -m "$MODEL" -c "$CTX" -np 1 -ngl 999 -fa on -lm "${LM:-mmap}" --jinja \
  --host 127.0.0.1 --port "$PORT" --no-webui "$@" > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 $PID 2>/dev/null; then echo "{\"label\":\"$LABEL\",\"error\":\"server died\"}" | tee -a "$HERE/results.jsonl"; tail -20 "$LOG" >&2; exit 1; fi
  sleep 1
done
EFF=(); [ -n "${EFFORT:-}" ] && EFF=(--effort "$EFFORT")
python3 -I "$HERE/bench_server.py" --port "$PORT" --label "$LABEL" \
  --prompts "${PROMPTS:-code,chat}" --reps "${REPS:-2}" --max-tokens "${MAXTOK:-512}" \
  --temp "${TEMP:-1.0}" --top-k "${TOPK:-20}" ${DUMP:+--dump "$DUMP"} "${EFF[@]}" | tee -a "$HERE/results.jsonl"
grep -E 'draft acceptance|graph splits|speculative checkpoint|model buffer size|KV buffer size|compute buffer size|CPU_Mapped' "$LOG" | tail -8 >&2
kill $PID; wait $PID 2>/dev/null
