import sys, json
label, outp = sys.argv[1], sys.argv[2]
with open(outp, "a") as out:
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line); r["label"] = label
        out.write(json.dumps(r) + "\n"); out.flush()
        t = "pp%d" % r["n_prompt"] if r["n_prompt"] else "tg%d" % r["n_gen"]
        print("%-28s ub=%5d d=%6d %-8s %9.2f +/- %.2f t/s" % (label, r["n_ubatch"], r["n_depth"], t, r["avg_ts"], r["stddev_ts"]), flush=True)
