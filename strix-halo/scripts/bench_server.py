#!/usr/bin/env python3
"""Hit a running llama-server with fixed prompts and print one JSON line per run.

Usage: bench_server.py --port 8099 --label NAME [--max-tokens 512] [--reps 2]
                       [--temp 1.0] [--top-p 0.95] [--top-k 20] [--effort high] [--prompts code,chat,long]
                       [--dump DIR]  (save each reply, for greedy output-equality checks)
Reads llama-server's own `timings` (prompt/predicted tok/s, draft stats).
"""
import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = {
    "code": "Write a complete, production-quality Python module implementing an LRU cache "
            "with O(1) get/put, TTL expiry, thread safety, type hints and docstrings, "
            "followed by a pytest test suite covering edge cases.",
    "chat": "Explain how TCP congestion control works: slow start, congestion avoidance, "
            "fast retransmit, fast recovery, and how CUBIC and BBR differ. Be thorough.",
}


def edit_prompt():
    with open(os.path.join(HERE, "edit_source.py"), encoding="utf-8") as f:
        src = f.read()
    return ("Here is a Python file:\n\n```python\n" + src + "```\n\n"
            "Rename the function `run_suite` to `run_all_tests` everywhere, and add a one-line docstring to "
            "`feedback`. Reply with the complete updated file in a single python code block and nothing else.")


def long_prompt():
    p = os.path.join(HERE, "long_context.txt")
    with open(p, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    return ("Here is a C++ source file:\n\n```cpp\n" + txt +
            "\n```\n\nSummarize what this file does, list its main functions, and point out "
            "two possible bugs or performance issues.")


def post(port, body, timeout=900):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--prompts", default="code,chat")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    for name in a.prompts.split(","):
        content = long_prompt() if name == "long" else edit_prompt() if name == "edit" else PROMPTS[name]
        for rep in range(a.reps):
            body = {
                "messages": [{"role": "user", "content": content}],
                "max_tokens": a.max_tokens, "temperature": a.temp,
                "top_p": a.top_p, "top_k": a.top_k, "min_p": 0.0, "seed": a.seed + rep,
                "cache_prompt": False,
            }
            if a.effort:
                body["chat_template_kwargs"] = {"reasoning_effort": a.effort}
            try:
                out, wall = post(a.port, body)
            except Exception as e:  # keep the sweep going
                print(json.dumps({"label": a.label, "prompt": name, "rep": rep, "error": str(e)}))
                sys.stdout.flush()
                continue
            t = out.get("timings", {})
            msg = out["choices"][0]["message"]
            rec = {
                "label": a.label, "prompt": name, "rep": rep,
                "pp_tok": t.get("prompt_n"), "pp_tps": round(t.get("prompt_per_second", 0), 1),
                "tg_tok": t.get("predicted_n"), "tg_tps": round(t.get("predicted_per_second", 0), 2),
                "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                "wall_s": round(wall, 2),
                "finish": out["choices"][0].get("finish_reason"),
                "reasoning_chars": len(msg.get("reasoning_content") or ""),
                "content_chars": len(msg.get("content") or ""),
            }
            if rec["draft_n"]:
                rec["acc_rate"] = round(rec["draft_acc"] / rec["draft_n"], 3)
            if a.dump:
                os.makedirs(a.dump, exist_ok=True)
                with open(os.path.join(a.dump, f"{a.label}-{name}-{rep}.txt"), "w") as fh:
                    fh.write((msg.get("reasoning_content") or "") + "\n=====\n" + (msg.get("content") or ""))
            print(json.dumps(rec)); sys.stdout.flush()


if __name__ == "__main__":
    main()
