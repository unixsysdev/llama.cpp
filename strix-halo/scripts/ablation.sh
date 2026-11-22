#!/bin/bash
# ablation.sh: prompt processing (4096-token batches) at several cache depths for
#   master  = upstream build
#   sparse  = this branch, coopmat indexer off   (sparse prefill attention only)
#   indexer = this branch, sparse prefill off    (coopmat lightning indexer only)
#   both    = this branch
# Pass 1 runs every config at all depths; pass 2 repeats the shallow depths in reverse order to cancel drift.
# Before each run it waits for a quiet machine (CPU and GPU share one power budget on Strix Halo), and it logs
# the GPU clock and temperature during the run to clocks.jsonl next to the results.
# Env: MODEL_DIR, BASE_BIN (upstream build-vulkan/bin), PATCHED_BIN (this branch), OUT (results dir), DEPTHS, DEPTHS2,
#      QUIET_LOAD (wait until the 1-minute load average is below this, default 6)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
M=${MODEL_DIR:-$HOME/models/qwen3.8-flash-next}/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
BASE=${BASE_BIN:?set BASE_BIN to an upstream build-vulkan/bin}
PATCHED=${PATCHED_BIN:-$HERE/../../build-vulkan/bin}
export OUT=${OUT:-$HERE/../results}
D1=${DEPTHS:-0,16384,32768,65536,131072}
D2=${DEPTHS2:-0,16384,32768}
CARD=$(ls -d /sys/class/drm/card*/device | head -1)
HWMON=$(ls -d "$CARD"/hwmon/hwmon* | head -1)
mkdir -p "$OUT"

wait_quiet() {
  while :; do
    l=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l="$l" -v q="${QUIET_LOAD:-6}" 'BEGIN { exit !(l < q) }' && break
    echo "waiting for a quiet machine (load $l)" >&2; sleep 30
  done
}

run() {  # run LABEL BIN DEPTHS [ENV=VAL]
  local label=$1 bin=$2 depths=$3; shift 3
  wait_quiet
  ( while :; do
      echo "$(date +%s) $(grep '\*' "$CARD/pp_dpm_sclk" | grep -oE '[0-9]+Mhz' | tr -d Mhz) $(( $(cat "$HWMON/temp1_input") / 1000 ))"
      sleep 2
    done ) > "$OUT/.clk.tmp" &
  local sampler=$!
  env "$@" "$HERE/lb.sh" "$label" "$bin" "$M" -p 4096 -n 0 -d "$depths" -b 4096 -ub 4096 -r 1
  kill $sampler
  python3 -I - "$OUT/.clk.tmp" "$label" "$(cut -d' ' -f1 /proc/loadavg)" >> "$OUT/clocks.jsonl" <<'EOF'
import json, statistics, sys
rows = [l.split() for l in open(sys.argv[1]) if len(l.split()) == 3]
clk = [int(r[1]) for r in rows]; tmp = [int(r[2]) for r in rows]
print(json.dumps({"label": sys.argv[2], "samples": len(rows), "sclk_median": statistics.median(clk) if clk else None,
                  "sclk_min": min(clk) if clk else None, "temp_max": max(tmp) if tmp else None, "load_after": float(sys.argv[3])}))
EOF
  rm -f "$OUT/.clk.tmp"
}

cd "$HERE"
for cfg in master sparse indexer both; do
  case $cfg in
    master)  run abl-master-r1  "$BASE"    "$D1" X=1 ;;
    sparse)  run abl-sparse-r1  "$PATCHED" "$D1" GGML_VK_DISABLE_LIGHTNING_INDEXER_COOPMAT=1 ;;
    indexer) run abl-indexer-r1 "$PATCHED" "$D1" GGML_VK_FA_SPARSE_PREFILL_DISABLE=1 ;;
    both)    run abl-both-r1    "$PATCHED" "$D1" X=1 ;;
  esac
done
for cfg in both indexer sparse master; do
  case $cfg in
    master)  run abl-master-r2  "$BASE"    "$D2" X=1 ;;
    sparse)  run abl-sparse-r2  "$PATCHED" "$D2" GGML_VK_DISABLE_LIGHTNING_INDEXER_COOPMAT=1 ;;
    indexer) run abl-indexer-r2 "$PATCHED" "$D2" GGML_VK_FA_SPARSE_PREFILL_DISABLE=1 ;;
    both)    run abl-both-r2    "$PATCHED" "$D2" X=1 ;;
  esac
done
