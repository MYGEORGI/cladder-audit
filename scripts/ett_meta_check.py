"""ETT on `confounding`: does the published ground truth equal the true ETT of the generator's SCM?

    python -B ett_meta_check.py <cladder csv> <cladder-v1-meta-models.json> <solver_check.json> <out json>

A task is linked to the models of its story whose parameters, rounded to two decimals as in the prompt, give the
prompt's numbers. True ETT = sum_v P(V1=v | X=1) [P(Y=1|v,1) - P(Y=1|v,0)], with P(V1|X=1) by Bayes from p(V1), p(X|V1).
"""

import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

csv_path, meta_path, solver_path, out_path = (Path(a) for a in sys.argv[1:5])
csv.field_size_limit(10**8)
rows = {r["id"]: r for r in csv.DictReader(csv_path.open(encoding="utf-8", newline=""))}
cases = {c["id"]: c for c in json.loads(solver_path.read_text(encoding="utf-8"))["cases"]}
models = [m for m in json.loads(meta_path.read_text(encoding="utf-8")) if m["graph_id"] == "confounding"]


def parsed(value):
    return value if isinstance(value, dict) else ast.literal_eval(value)  # the file mixes dicts and their repr


def numbers(m):
    p = parsed(m["params"])
    pv, px, py = p["p(V1)"], p["p(X | V1)"], p["p(Y | V1, X)"]
    pv = pv[0] if isinstance(pv, list) else pv  # root parameters are sometimes a one-element list (cladder issue #16)
    return pv, px, py


def true_ett(pv, px, py):
    w = [(1 - pv) * px[0], pv * px[1]]  # P(V1=v, X=1)
    return sum(w[v] / sum(w) * (py[v][1] - py[v][0]) for v in (0, 1))


def ate(pv, px, py):
    return sum((pv if v else 1 - pv) * (py[v][1] - py[v][0]) for v in (0, 1))


result, verdicts = {}, Counter()
for tid, row in rows.items():
    if row["query_type"] != "ett" or row["graph_id"] != "confounding":
        continue
    r = {(int(a), int(b)): float(c)
         for a, b, c in re.findall(r"P\(Y=1 \| V1=(\d), X=(\d)\) = ([0-9.]+)", row["reasoning"])}
    pv_txt = float(re.search(r"P\(V1=1\) = ([0-9.]+)", row["reasoning"])[1])
    hits = []
    for m in models:
        if m["story_id"] != row["story_id"]:
            continue
        pv, px, py = numbers(m)
        if round(pv, 2) == pv_txt and all(round(py[v][x], 2) == r[(v, x)] for v in (0, 1) for x in (0, 1)):
            gt = parsed(m["groundtruth"])["ETT(Y | X)"]
            hits.append({"model_id": m["model_id"], "gt_ett": gt, "true_ett": true_ett(pv, px, py), "ate": ate(pv, px, py)})
    category = cases[tid]["category"]
    for h in hits:
        h["gt_is"] = ("true_ett" if abs(h["gt_ett"] - h["true_ett"]) < 1e-9
                      else "ate" if abs(h["gt_ett"] - h["ate"]) < 1e-9 else "other")
        h["sign_flip"] = (h["gt_ett"] > 0) != (h["true_ett"] > 0)
    # several models can round to the same prompt numbers; the verdict stands only if they all agree
    kinds = {(h["gt_is"], h["sign_flip"], h["gt_ett"] > 0) for h in hits}
    if len(kinds) != 1:
        verdicts[(category, f"{len(hits)} models, disagree")] += 1
    else:
        gt_is, flip, _ = kinds.pop()
        verdicts[(category, f"gt={gt_is}", "sign flips" if flip else "same sign")] += 1
    result[tid] = {"category": category, "models": hits}
out_path.write_text(json.dumps({"verdicts": {" | ".join(k): v for k, v in sorted(verdicts.items())}, "tasks": result},
                               indent=1) + "\n", encoding="utf-8", newline="\n")
for k, v in sorted(verdicts.items()):
    print(v, k)
