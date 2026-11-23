#!/usr/bin/env python3
"""profile_shapes.py BIN_DIR MODEL DEPTH OUT_TXT [ENV=VAL ...]

GPU time per op of one 4096-token prompt batch at DEPTH, from GGML_VK_PERF_LOGGER=1, with the matmul shapes kept
(profile.py merges them by type). Writes the top 45 ops to OUT_TXT.
"""
import collections, os, re, subprocess, sys

bindir, model, depth, out = sys.argv[1:5]
env = dict(a.split("=", 1) for a in sys.argv[5:])
cmd = [os.path.join(bindir, "llama-bench"), "-m", model, "-ngl", "999", "-fa", "on", "-lm", "mmap", "-p", "4096", "-n", "0",
       "-d", depth, "-b", "4096", "-ub", "4096", "-r", "1", "-o", "jsonl"]
p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env, "GGML_VK_PERF_LOGGER": "1"})
block = p.stderr.split("Vulkan Timings:")[-1]
ops, calls = collections.Counter(), collections.Counter()
for line in block.splitlines():
    m = re.match(r"(.*?):\s+(\d+) x ([\d.]+) us = ([\d.e+]+) us", line)
    if m:
        ops[m.group(1).strip()] += float(m.group(4)) / 1000
        calls[m.group(1).strip()] += int(m.group(2))
total = sum(ops.values())
with open(out, "w") as f:
    f.write(f"depth {depth}: {total:.0f} ms GPU per 4096-token batch\n")
    for k, v in ops.most_common(45):
        f.write(f"{v:8.1f} ms {100 * v / total:5.1f}% {calls[k]:5d}x  {k}\n")
print(open(out).read())
