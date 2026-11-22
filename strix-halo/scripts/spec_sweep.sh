#!/bin/bash
# spec_sweep.sh LABEL BIN_DIR CONFIG... ; CONFIG = label:"server args" ; runs run_server_bench.sh for each (prompts x REPS)
# MTP and DFLASH in the args expand to the sidecar paths in MODEL_DIR. Env: MODEL_DIR, plus run_server_bench.sh's.
HERE=$(cd "$(dirname "$0")" && pwd); TREE=$1; BIN=$2; shift 2
Q=${MODEL_DIR:-$HOME/models/qwen3.8-flash-next}
M=$Q/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
for c in "$@"; do
  label=${c%%:*}; args=${c#*:}; args=${args//MTP/$Q/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}; args=${args//DFLASH/$Q/dflash-Qwen3.8-Flash-Next-Q8_0.gguf}
  echo "### $TREE-$label: $args"
  "$HERE/run_server_bench.sh" "$TREE-$label" "$BIN" "$M" $args -lv 3 2>&1 | grep -vE '^$'
done
