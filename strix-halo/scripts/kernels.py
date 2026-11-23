#!/usr/bin/env python3
"""kernels.py TEST_BACKEND_OPS OUT_JSON -> kernel timings for the two changes, each with the change on and off.

Runs `test-backend-ops perf` on Vulkan0 for
  - FLASH_ATTN_EXT, Qwen QSA shape (hs 256, gqa 12, 512 tokens, n_kv_max 2051) at 8K..131K cache,
    with sparse prefill on and off (GGML_VK_FA_SPARSE_PREFILL_DISABLE),
  - LIGHTNING_INDEXER, 4 heads x 128, f16 keys, with the coopmat kernel on and off
    (GGML_VK_DISABLE_LIGHTNING_INDEXER_COOPMAT).
"""
import json, os, re, subprocess, sys

exe, out = sys.argv[1], sys.argv[2]


def perf(op, pattern, env):
    e = {**os.environ, **env}
    p = subprocess.run([exe, "perf", "-b", "Vulkan0", "-o", op, "-p", pattern], capture_output=True, text=True, env=e,
                       timeout=1800)
    res = {}
    for line in p.stdout.splitlines():
        m = re.search(r"\((.*)\):\s+\d+ runs -\s+([\d.]+) us/run", line)
        if m:
            res[m.group(1)] = float(m.group(2))
    return res


def field(params, name):
    return int(re.search(rf"(?:^|,){name}=(\d+)", params).group(1))


fa_pat = "nb=512.*n_kv_max=2051"
fa_on = perf("FLASH_ATTN_EXT", fa_pat, {"GGML_VK_FA_SPARSE_PREFILL_RATIO": "1"})
fa_off = perf("FLASH_ATTN_EXT", fa_pat, {"GGML_VK_FA_SPARSE_PREFILL_DISABLE": "1"})
attention = sorted(({"kv": field(k, "kv"), "dense_us": fa_off[k], "sparse_us": v} for k, v in fa_on.items() if k in fa_off),
                   key=lambda r: r["kv"])

li_pat = "nh=4,kv=(4096|65536),nb=(1|512|2048),ns=1,nm=1,type_K=f16,w_const=0"
li_on = perf("LIGHTNING_INDEXER", li_pat, {})
li_off = perf("LIGHTNING_INDEXER", li_pat, {"GGML_VK_DISABLE_LIGHTNING_INDEXER_COOPMAT": "1"})
indexer = sorted(({"kv": field(k, "kv"), "nb": field(k, "nb"), "scalar_us": li_off[k], "coopmat_us": v}
                  for k, v in li_on.items() if k in li_off and field(k, "nb") > 1), key=lambda r: (r["kv"], r["nb"]))
decode = [{"kv": field(k, "kv"), "nb": 1, "scalar_us": li_off[k], "selected_us": v}
          for k, v in li_on.items() if k in li_off and field(k, "nb") == 1]

json.dump({"attention": attention, "indexer": indexer, "indexer_decode": decode}, open(out, "w"), indent=1)
print(json.dumps({"attention": attention, "indexer": indexer, "indexer_decode": decode}, indent=1))
