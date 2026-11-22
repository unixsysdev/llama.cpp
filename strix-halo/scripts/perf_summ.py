#!/usr/bin/env python3
"""perf_summ.py ERRFILE [BLOCK] -> totals per op group from GGML_VK_PERF_LOGGER output (last block by default)."""
import re, sys, collections
txt = open(sys.argv[1]).read().split("Vulkan Timings:")[1:]
blk = txt[int(sys.argv[2]) if len(sys.argv) > 2 else -1]
rows = []
for line in blk.splitlines():
    m = re.match(r"(.*?):\s+(\d+) x ([\d.]+) us = ([\d.e+]+) us", line)
    if m: rows.append((m.group(1), int(m.group(2)), float(m.group(4))))
tot = sum(r[2] for r in rows)
grp = collections.Counter()
for name, n, t in rows:
    key = name.split(" m=")[0] if name.startswith("MUL_MAT") else name.split(" dst")[0].split("(")[0]
    grp[key] += t
print(f"{len(txt)} blocks; this block total {tot/1000:.1f} ms")
for k, v in grp.most_common(25): print(f"{v/1000:8.1f} ms {100*v/tot:5.1f}%  {k}")
