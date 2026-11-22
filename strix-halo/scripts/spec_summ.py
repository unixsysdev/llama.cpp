#!/usr/bin/env python3
"""spec_summ.py SWEEP_LOG -> per-config averages (tg t/s, acceptance) from spec_sweep.sh output."""
import sys, json, collections
cur, rows = None, collections.OrderedDict()
for l in open(sys.argv[1]):
    if l.startswith("###"): cur = l[4:].split(":")[0]; rows[cur] = []; continue
    if l.startswith('{"label"'):
        r = json.loads(l)
        if "error" in r: rows[cur].append(None)
        else: rows[cur].append(r)
for k, rs in rows.items():
    ok = [r for r in rs if r]
    if not ok: print(f"{k:28s} ERROR/none"); continue
    tg = sum(r["tg_tps"] for r in ok) / len(ok)
    by = collections.defaultdict(list)
    for r in ok: by[r["prompt"]].append(r["tg_tps"])
    dn = sum(r["draft_n"] or 0 for r in ok); da = sum(r["draft_acc"] or 0 for r in ok)
    acc = f"{da/dn:.2f}" if dn else "-"
    per = "  ".join(f"{p} {sum(v)/len(v):5.1f}" for p, v in by.items())
    print(f"{k:28s} tg {tg:5.1f} t/s  ({per})  acc {acc}  n={len(ok)}")
