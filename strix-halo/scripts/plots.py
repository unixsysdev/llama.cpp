#!/usr/bin/env python3
"""plots.py [RESULTS_DIR] [FIGURES_DIR] -> the figures used in README.md and REPORT.md.

Reads only the files in results/ (llama-bench JSONL, server benchmark JSONL, depth curves, kernel and
profile tables), so every figure can be regenerated from the published raw data.
"""
import collections
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
FIG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "figures")

# light-surface tokens and the categorical order (fixed, never cycled)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5, "axes.titlesize": 12.5,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 7, "figure.dpi": 150,
})


def load_jsonl(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def kfmt(x):
    return f"{int(x / 1024)}K" if x >= 1024 else str(int(x))


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=9.5,
                color=INK2, fontweight="semibold")
    ax.plot([x], [y], "o", color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)


def fig_prefill_vs_depth():
    rows = []
    for r in load_jsonl("llama-bench.jsonl"):
        if r.get("label", "").startswith("abl-") and r["n_prompt"]:
            rows.append({**r, "label": r["label"].rsplit("-r", 1)[0]})  # abl-<cfg>-r<rep> -> abl-<cfg>
    if not rows:
        return
    names = [("abl-master", "upstream master"), ("abl-indexer", "+ coopmat indexer"),
             ("abl-sparse", "+ sparse prefill attention"), ("abl-both", "both (this branch)")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    offsets = {"abl-master": -3, "abl-indexer": 3, "abl-sparse": 0, "abl-both": 0}  # keep the end labels apart
    for i, (lab, title) in enumerate(names):
        by_depth = collections.defaultdict(list)
        for r in rows:
            if r["label"] == lab:
                by_depth[r["n_depth"]].append(r["avg_ts"])
        pts = sorted((d, sum(v) / len(v)) for d, v in by_depth.items())  # mean over repetitions
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=S[i], label=title, markeredgecolor=SURFACE, markeredgewidth=1.5)
        end_label(ax, xs[-1], ys[-1], f"{ys[-1]:.0f}", S[i], offsets[lab] * 3)
    ax.set_xticks(sorted({r["n_depth"] for r in rows}))
    ax.set_xticklabels([kfmt(x) for x in sorted({r["n_depth"] for r in rows})])
    ax.set_xlabel("context already in the cache (tokens)")
    ax.set_ylabel("prompt processing (tokens/s)")
    ax.set_ylim(0, None)
    ax.set_title("Prompt processing vs context depth (4096-token batches)")
    ax.legend(loc="lower left", ncol=2)
    save(fig, "prefill_vs_depth.png")


def fig_depth_curve():
    curves = [("depth-curve-master.jsonl", "upstream master", S[0]), ("depth-curve-final.jsonl", "this branch", S[3])]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    any_ = False
    for fname, title, color in curves:
        recs = [r for r in load_jsonl(fname) if r.get("turn") == 1]
        if not recs:
            continue
        r = recs[0]
        xs, ys = zip(*r["curve"][:-1]) if len(r["curve"]) > 1 else zip(*r["curve"])
        mins = r["prompt_s"] / 60
        ax.plot(xs, ys, "-", color=color, label=f"{title}: {r['prompt_n'] / 1000:.0f}K tokens in {mins:.1f} min")
        end_label(ax, xs[-1], ys[-1], f"{ys[-1]:.0f}", color)
        any_ = True
    if not any_:
        plt.close(fig)
        return
    ticks = [0, 65536, 131072, 196608, 262144]
    ax.set_xticks(ticks)
    ax.set_xticklabels([kfmt(t) for t in ticks])
    ax.set_xlim(0, 262144)
    ax.set_xlabel("position in the prompt (tokens)")
    ax.set_ylabel("prompt processing (tokens/s, per 16K chunk)")
    ax.set_ylim(0, None)
    ax.set_title("One 250K-token prompt, end to end (llama-server, 256K context)")
    ax.legend(loc="lower left")
    save(fig, "depth_curve_250k.png")


def server_rows():
    return load_jsonl("server-bench.jsonl")


def avg_by_label(rows, labels, prompt=None):
    out = collections.OrderedDict()
    for lab in labels:
        rs = [r for r in rows if r.get("label") == lab and "tg_tps" in r and (prompt is None or r["prompt"] == prompt)]
        if rs:
            dn = sum(r.get("draft_n") or 0 for r in rs)
            da = sum(r.get("draft_acc") or 0 for r in rs)
            out[lab] = (sum(r["tg_tps"] for r in rs) / len(rs), da / dn if dn else None)
    return out


def fig_decode_spec():
    rows = server_rows()
    cfg = [("master-plain", "no drafting"), ("master-mtp3", "MTP 3, greedy"), ("master-mtp2", "MTP 2, greedy"),
           ("master-mtp4", "MTP 4, greedy"), ("master-dflash2-n4prob", "DFlash2 4, probabilistic"),
           ("master-mtp3prob", "MTP 3, probabilistic"), ("master-mtp4prob", "MTP 4, probabilistic"),
           ("master-mtp2prob", "MTP 2, probabilistic")]
    vals = avg_by_label(rows, [c[0] for c in cfg])
    items = [(t, *vals[l]) for l, t in cfg if l in vals]
    if not items:
        return
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ys = range(len(items))
    ax.barh(ys, [v for _, v, _ in items], height=0.62, color=S[0], edgecolor=SURFACE, linewidth=2)
    for y, (t, v, acc) in zip(ys, items):
        ax.text(v + 0.4, y, f"{v:.1f}" + (f"   ({acc * 100:.0f}% accepted)" if acc else ""), va="center",
                fontsize=9.5, color=INK2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([t for t, _, _ in items])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v for _, v, _ in items) * 1.35)
    ax.set_xlabel("decode (tokens/s), thinking on, effort xhigh, T=1.0, avg of code + chat prompts")
    ax.set_title("Speculative decoding: draft length and draft sampling")
    save(fig, "decode_speculative.png")


def fig_decode_low_effort():
    rows = server_rows()
    cfg = [("master-lo-ngram", "n-gram only"), ("master-lo-mtp1prob", "MTP 1"), ("master-lo-mtp2prob", "MTP 2"),
           ("master-lo-mtp2prob-ngram", "MTP 2 + n-gram")]
    code = avg_by_label(rows, [c[0] for c in cfg], "code")
    edit = avg_by_label(rows, [c[0] for c in cfg], "edit")
    labs = [c for c in cfg if c[0] in code and c[0] in edit]
    if not labs:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    w = 0.38
    for j, (series, name) in enumerate(((code, "writing new code"), (edit, "editing a file (re-emits it)"))):
        xs = [i + (j - 0.5) * (w + 0.02) for i in range(len(labs))]
        vs = [series[l][0] for l, _ in labs]
        ax.bar(xs, vs, width=w, color=S[j], label=name, edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vs):
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=9.5, color=INK2)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels([t for _, t in labs])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("decode (tokens/s)")
    ax.set_ylim(0, 125)
    ax.set_title("Decode at reasoning effort low (probabilistic drafts)")
    ax.legend(loc="upper center", ncol=2)
    save(fig, "decode_low_effort.png")


def fig_kernels():
    path = os.path.join(RES, "kernels.json")
    if not os.path.exists(path):
        return
    k = json.load(open(path))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    kv = [r["kv"] for r in k["attention"]]
    for i, (key, name) in enumerate((("dense_us", "dense (upstream)"), ("sparse_us", "sparse prefill"))):
        ys = [r[key] / 1000 for r in k["attention"]]
        a1.plot(kv, ys, "-o", color=S[i], label=name, markeredgecolor=SURFACE, markeredgewidth=1.5)
        end_label(a1, kv[-1], ys[-1], f"{ys[-1]:.1f} ms", S[i])
    a1.set_xscale("log", base=2)
    a1.set_xticks(kv)
    a1.set_xticklabels([kfmt(x) for x in kv])
    a1.set_xlabel("cache size (tokens)")
    a1.set_ylabel("ms per call")
    a1.set_title("QSA attention, 512-token batch")
    a1.legend(loc="upper left")
    shapes = [f"{kfmt(r['kv'])} keys\n{r['nb']} tok" for r in k["indexer"]]
    w = 0.38
    for j, (key, name) in enumerate((("scalar_us", "scalar (upstream)"), ("coopmat_us", "coopmat"))):
        xs = [i + (j - 0.5) * (w + 0.02) for i in range(len(shapes))]
        vs = [r[key] / 1000 for r in k["indexer"]]
        a2.bar(xs, vs, width=w, color=S[j], label=name, edgecolor=SURFACE, linewidth=2)
    for i, r in enumerate(k["indexer"]):
        a2.text(i, max(r["scalar_us"], r["coopmat_us"]) / 1000 * 1.02 + 0.5,
                f"{r['scalar_us'] / r['coopmat_us']:.1f}x", ha="center", fontsize=9.5, color=INK2, fontweight="semibold")
    a2.set_xticks(range(len(shapes)))
    a2.set_xticklabels(shapes, fontsize=9)
    a2.grid(axis="x", visible=False)
    a2.set_ylabel("ms per call")
    a2.set_title("Lightning indexer, 4 heads x 128")
    a2.legend(loc="upper left")
    fig.tight_layout(w_pad=3)
    save(fig, "kernels.png")


def fig_profile():
    path = os.path.join(RES, "profile_131k.json")
    if not os.path.exists(path):
        return
    p = json.load(open(path))
    groups = ["attention", "indexer + top-k + mask", "dense matmul", "expert matmul", "other"]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    runs = list(p.keys())
    for y, run in enumerate(runs):
        left = 0
        for i, g in enumerate(groups):
            v = p[run].get(g, 0) / 1000
            ax.barh(y, v, left=left, height=0.55, color=S[i], edgecolor=SURFACE, linewidth=2,
                    label=g if y == 0 else None)
            left += v
        ax.text(left + 0.1, y, f"{left:.1f} s", va="center", fontsize=9.5, color=INK2, fontweight="semibold")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("GPU time per 4096-token batch with 128K already cached (s)")
    ax.set_title("Where the prompt-processing time goes at 128K")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=5, fontsize=9)
    save(fig, "profile_131k.png")


def fig_ubatch():
    rows = load_jsonl("llama-bench.jsonl")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    any_ = False
    for i, (lab, name) in enumerate((("vk-master", "Vulkan (RADV)"), ("hip-master-lmnone", "ROCm/HIP"))):
        pts = sorted({(r["n_ubatch"], r["avg_ts"]) for r in rows if r.get("label") == lab and r["n_prompt"] == 4096})
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=S[i], label=name, markeredgecolor=SURFACE, markeredgewidth=1.5)
        end_label(ax, xs[-1], ys[-1], f"{ys[-1]:.0f}", S[i])
        any_ = True
    if not any_:
        plt.close(fig)
        return
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["512", "1K", "2K", "4K", "8K"])
    ax.set_ylim(0, None)
    ax.set_xlabel("physical batch size (-ub)")
    ax.set_ylabel("prompt processing (tokens/s)")
    ax.set_title("Batch size, 4096-token prompt, empty cache")
    ax.legend(loc="lower right")
    save(fig, "ubatch.png")


if __name__ == "__main__":
    fig_prefill_vs_depth()
    fig_depth_curve()
    fig_decode_spec()
    fig_decode_low_effort()
    fig_kernels()
    fig_profile()
    fig_ubatch()
