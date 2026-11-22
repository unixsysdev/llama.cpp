#!/usr/bin/env python3
"""Multi-turn coding session against a local llama-server (OpenAI API).

The model builds the `calc` package stage by stage (spec/stageN.md + tests/test_stageN.py shown
to it). After every reply the files are written to work/, committed to git, and all visible and
hidden tests up to the current stage are run. Failures go back to the model (up to MAX_FIX times).
Logs: transcript.jsonl (full messages), metrics.jsonl (one line per turn), progress on stdout.
"""
import json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
URL = os.environ.get("URL", "http://127.0.0.1:8081/v1/chat/completions")
MAX_FIX = int(os.environ.get("MAX_FIX", "2"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16384"))
STAGES = range(int(os.environ.get("FIRST", "1")), int(os.environ.get("LAST", "7")) + 1)

SYSTEM = """You are a senior Python engineer pair-programming with me. We are building `calc`, an interpreter for a small language, as a Python package in the directory `calc/` (Python 3.12+, standard library only).

How to reply:
- For every file you create or change, output its COMPLETE new content: a line `FILE: calc/<name>.py` followed by the code in a ```python fenced block. Never send partial files, diffs or placeholders like "rest unchanged". Files you don't mention stay as they are.
- Don't write test files; I run the tests myself and will tell you about failures.
- After the code, add a short note (a few lines) about what you changed and any design decisions.
- Keep the design clean and easy to extend: we will keep adding features."""


EFFORT = os.environ.get("EFFORT", "")


def chat(messages):
    body = {"model": "inkling-2-mini", "messages": messages, "max_tokens": MAX_TOKENS}
    if EFFORT:
        body["chat_template_kwargs"] = {"reasoning_effort": EFFORT}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out, time.time() - t0


FILE_RE = re.compile(r"^[#*>\s`]*FILE:\s*`?([\w./-]+?)`?\**\s*\n+```[a-zA-Z0-9]*\n(.*?)\n```", re.M | re.S)


def extract_files(content):
    files = {}
    for path, code in FILE_RE.findall(content or ""):
        norm = os.path.normpath(path)
        if norm.startswith("calc" + os.sep) and norm.endswith(".py") and ".." not in norm.split(os.sep):
            files[norm] = code + "\n"
    return files


def git(*args):
    subprocess.run(["git", "-C", WORK, *args], capture_output=True, text=True)


def count_tests(path):
    with open(path) as f:
        return len(re.findall(r"^\s+def test_", f.read(), re.M))


def run_suite(stage):
    res = {"visible": [0, 0], "hidden": [0, 0], "failures": []}
    for kind, d, suffix in (("visible", "tests", ""), ("hidden", "hidden", "_hidden")):
        for s in range(1, stage + 1):
            tf = os.path.join(HERE, d, f"test_stage{s}{suffix}.py")
            total = count_tests(tf)
            try:
                p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_tests.py"), WORK, tf],
                                   capture_output=True, text=True, timeout=180)
                r = json.loads(p.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                r = {"passed": 0, "failures": [{"test": "(whole file)", "kind": "error", "detail": "timed out after 180 s"}]}
            except Exception as e:  # noqa: BLE001
                r = {"passed": 0, "failures": [{"test": "(runner)", "kind": "error", "detail": repr(e)}]}
            res[kind][0] += min(r["passed"], total)
            res[kind][1] += total
            for f in r["failures"]:
                res["failures"].append({**f, "where": "tests/test_stage%d.py" % s if kind == "visible" else "my extra checks (stage %d)" % s})
    return res


def feedback(res, missing_files, finish):
    if missing_files:
        msg = "I couldn't find any files in your reply"
        if finish == "length":
            msg += " (it was cut off at the length limit)"
        return msg + ". Please send the complete contents of every file you need, each preceded by `FILE: calc/<name>.py`."
    lines = [f"I ran the tests: {res['visible'][0]}/{res['visible'][1]} of the visible tests pass, "
             f"and {res['hidden'][0]}/{res['hidden'][1]} of my extra checks pass. Failures:"]
    budget = 6000
    for f in res["failures"]:
        item = f"\n--- {f['test']} ({f['where']}, {f['kind']}):\n{f['detail']}"
        if budget - len(item) < 0:
            lines.append(f"\n... and {len(res['failures'])} failures in total; fix the ones above first.")
            break
        lines.append(item)
        budget -= len(item)
    lines.append("\n\nPlease fix these. Remember to send complete files.")
    return "".join(lines)


def log(path, rec):
    with open(os.path.join(HERE, path), "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    os.makedirs(os.path.join(WORK, "calc"), exist_ok=True)
    git("init", "-q"); git("config", "user.name", "inkling"); git("config", "user.email", "inkling@localhost")
    messages = [{"role": "system", "content": SYSTEM}]
    if os.environ.get("RESUME") and os.path.exists(os.path.join(HERE, "transcript.jsonl")):
        # rebuild the conversation so far, byte for byte, to continue the same session
        for line in open(os.path.join(HERE, "transcript.jsonl")):
            t = json.loads(line)
            messages.append({"role": "user", "content": t["user"]})
            messages.append({"role": "assistant", "content": t["content"], "reasoning_content": t["reasoning"]})
        print(f"resumed {len(messages) // 2} turns, effort={EFFORT or 'default'}", flush=True)
    t_start = time.time()
    for stage in STAGES:
        with open(os.path.join(HERE, "spec", f"stage{stage}.md")) as f:
            spec = f.read()
        with open(os.path.join(HERE, "tests", f"test_stage{stage}.py")) as f:
            tests = f.read()
        user = (spec + f"\n\nThese tests must pass (`tests/test_stage{stage}.py`), and all tests from earlier steps must keep passing:\n\n```python\n{tests}```")
        for attempt in range(MAX_FIX + 1):
            messages.append({"role": "user", "content": user})
            out, wall = chat(messages)
            choice = out["choices"][0]
            msg = choice["message"]
            content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
            messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning})
            log("transcript.jsonl", {"stage": stage, "attempt": attempt, "user": user, "reasoning": reasoning, "content": content})
            files = extract_files(content)
            for path, code in files.items():
                full = os.path.join(WORK, path)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as fh:
                    fh.write(code)
            git("add", "-A"); git("commit", "-q", "-m", f"stage {stage} attempt {attempt}")
            res = run_suite(stage)
            t = out.get("timings", {})
            rec = {"stage": stage, "attempt": attempt, "finish": choice.get("finish_reason"),
                   "ctx_after": (t.get("cache_n") or 0) + (t.get("prompt_n") or 0) + (t.get("predicted_n") or 0),
                   "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"), "prompt_s": round((t.get("prompt_ms") or 0) / 1000, 1),
                   "gen_tok": t.get("predicted_n"), "tg_tps": round(t.get("predicted_per_second") or 0, 1),
                   "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "wall_s": round(wall, 1),
                   "reasoning_chars": len(reasoning), "content_chars": len(content), "files": sorted(files),
                   "visible": res["visible"], "hidden": res["hidden"], "elapsed_min": round((time.time() - t_start) / 60, 1)}
            log("metrics.jsonl", rec)
            acc = f"{rec['draft_acc'] / rec['draft_n']:.2f}" if rec["draft_n"] else "-"
            print(f"stage {stage} try {attempt}: ctx {rec['ctx_after']:6d} | new prompt {rec['prompt_n']:5d} in {rec['prompt_s']:5.1f}s | "
                  f"gen {rec['gen_tok']:5d} @ {rec['tg_tps']:4.1f} t/s (acc {acc}) | wall {rec['wall_s']:6.1f}s | "
                  f"visible {res['visible'][0]}/{res['visible'][1]} hidden {res['hidden'][0]}/{res['hidden'][1]} | "
                  f"files {','.join(os.path.basename(p) for p in files) or 'NONE'} | {rec['finish']}", flush=True)
            if res["visible"][0] == res["visible"][1] and res["hidden"][0] == res["hidden"][1]:
                break
            user = feedback(res, not files, choice.get("finish_reason"))
    print(f"DONE in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
