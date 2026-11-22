#!/usr/bin/env python3
"""depth_test.py --port P --tokens N [--followup 2000] [--max-tokens 512] [--effort low]

Two-turn long-context test against a running llama-server:
  turn 1: ~N tokens of C++ source (code_1300k.txt) + a question -> prompt speed over the whole prompt, decode at depth
  turn 2: the same conversation + ~F new tokens + a question -> prefix-cache reuse and prompt speed of the new chunk at depth
Prints one JSON line per turn. Pair with the server log ("prompt processing progress") for the speed-vs-depth curve.
"""
import argparse, json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def call(port, path, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t0


def stream_chat(port, body, timeout=7200):
    """Streamed chat completion with prompt progress. Returns (message, timings, progress points, wall)."""
    body = {**body, "stream": True, "return_progress": True, "timings_per_token": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    content, reasoning, timings, prog = [], [], {}, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if "prompt_progress" in ev:
                pp = ev["prompt_progress"]
                prog.append((pp["cache"] + pp["processed"], pp["time_ms"], pp["total"]))
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content"): content.append(d["content"])
                if d.get("reasoning_content"): reasoning.append(d["reasoning_content"])
    msg = {"content": "".join(content), "reasoning_content": "".join(reasoning)}
    return msg, timings, prog, time.time() - t0


def take_tokens(port, text, n):
    """Longest prefix of text (cut at a line end) with about n tokens."""
    toks = call(port, "/tokenize", {"content": text})[0]["tokens"]
    if len(toks) <= n:
        return text, len(toks)
    cut = call(port, "/detokenize", {"tokens": toks[:n]})[0]["content"]
    cut = cut[: cut.rfind("\n") + 1]
    return cut, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--tokens", type=int, default=250000)
    ap.add_argument("--followup", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--label", default="depth")
    ap.add_argument("--step", type=int, default=16384, help="depth bucket for the speed curve")
    a = ap.parse_args()

    with open(os.path.join(HERE, "code_1300k.txt"), encoding="utf-8", errors="replace") as f:
        corpus = f.read()
    body_txt, n1 = take_tokens(a.port, corpus, a.tokens)
    rest = corpus[len(body_txt):]
    new_txt, n2 = take_tokens(a.port, rest, a.followup)

    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": a.max_tokens,
                "chat_template_kwargs": {"reasoning_effort": a.effort}}
    msgs = [{"role": "user", "content": "Here is part of a C++ code base:\n\n```cpp\n" + body_txt +
             "```\n\nWhich file defines the KV cache, and what are its three most important methods? Be brief."}]
    for turn in (1, 2):
        m, t, prog, wall = stream_chat(a.port, {"messages": msgs, **sampling})
        curve, last = [], (prog[0][0], 0.0) if prog else None
        for depth, ms, total in prog[1:]:
            if depth - last[0] >= a.step or depth == total:
                if ms > last[1]:
                    curve.append([depth, round((depth - last[0]) / ((ms - last[1]) / 1000), 1)])
                last = (depth, ms)
        rec = {"label": a.label, "turn": turn, "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
               "prompt_s": round(t.get("prompt_ms", 0) / 1000, 1), "pp_tps": round(t.get("prompt_per_second", 0), 1),
               "gen": t.get("predicted_n"), "tg_tps": round(t.get("predicted_per_second", 0), 2),
               "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "wall_s": round(wall, 1),
               "curve": curve, "answer": (m.get("content") or "")[:200]}
        print(json.dumps(rec), flush=True)
        msgs.append({"role": "assistant", "content": m.get("content") or "",
                     "reasoning_content": m.get("reasoning_content") or ""})
        msgs.append({"role": "user", "content": "Here is more of the code base:\n\n```cpp\n" + new_txt +
                     "```\n\nWhat does this new part add? One paragraph."})


if __name__ == "__main__":
    main()
