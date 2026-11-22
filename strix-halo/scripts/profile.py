#!/usr/bin/env python3
"""profile.py OUT_JSON NAME BIN_DIR MODEL [DEPTH] [ENV=VAL ...]

Runs one 4096-token prompt batch at DEPTH (default 131072) under GGML_VK_PERF_LOGGER=1 and adds the GPU time of the
last graph into op groups. Merges {NAME: {group: ms}} into OUT_JSON and prints the per-op table.
"""
import collections, json, os, re, subprocess, sys

out, name, bindir, model = sys.argv[1:5]
depth = sys.argv[5] if len(sys.argv) > 5 and "=" not in sys.argv[5] else "131072"
env = dict(a.split("=", 1) for a in sys.argv[5:] if "=" in a)

cmd = [os.path.join(bindir, "llama-bench"), "-m", model, "-ngl", "999", "-fa", "on", "-lm", "mmap", "-p", "4096", "-n", "0",
       "-d", depth, "-b", "4096", "-ub", "4096", "-r", "1"]
p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env, "GGML_VK_PERF_LOGGER": "1"})
block = p.stderr.split("Vulkan Timings:")[-1]


def group(op):
    if op.startswith("FLASH_ATTN_EXT"):
        return "attention"
    if op.startswith(("LIGHTNING_INDEXER", "TOP_K")):
        return "indexer + top-k + mask"
    if op.startswith("MUL_MAT_ID"):
        return "expert matmul"
    if op.startswith("MUL_MAT"):
        return "dense matmul"
    return "other"


g, per_op = collections.Counter(), collections.Counter()
for line in block.splitlines():
    m = re.match(r"(.*?):\s+(\d+) x ([\d.]+) us = ([\d.e+]+) us", line)
    if m:
        op = m.group(1)
        key = op.split(" m=")[0] if op.startswith("MUL_MAT") else op.split(" dst")[0].split("(")[0].strip()
        g[group(op)] += float(m.group(4)) / 1000
        per_op[key] += float(m.group(4)) / 1000

data = json.load(open(out)) if os.path.exists(out) else {}
data[name] = {k: round(v, 1) for k, v in g.items()}
json.dump(data, open(out, "w"), indent=1)
total = sum(per_op.values())
print(f"{name}: {total:.0f} ms GPU per 4096-token batch at depth {depth}")
for k, v in per_op.most_common(20):
    print(f"  {v:8.1f} ms {100 * v / total:5.1f}%  {k}")
