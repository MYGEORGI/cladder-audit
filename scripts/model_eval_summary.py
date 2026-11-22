"""Accuracy of the evaluated models on the subsets of the audit.

    python -B scripts/model_eval_summary.py [--out results/model_eval_summary.json]

Subsets (results/label_readings.csv, results/single_model.json): all tasks; solvable under reading R; text-undetermined
under reading R; the certified core of the undetermined tasks (both answers realized by one Bayesian network, or a
missing term, or a zero denominator); the 26 contradictions. For each model and subset: tasks, correct (answer equals
the label), invalid replies, accuracy with a 95% Wilson interval. On the undetermined tasks with a literal answer (reading
E decides), the share of replies equal to the literal answer measures how far a model follows the printed numbers.
"""

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "model_eval_summary.json")
    args = ap.parse_args()
    readings = {r["id"]: r for r in csv.DictReader((ROOT / "results" / "label_readings.csv").open(encoding="utf-8"))}
    single = json.loads((ROOT / "results" / "single_model.json").read_text(encoding="utf-8"))["items"]
    core = {i for i, r in readings.items() if r["reading_R"] == "undecidable"
            and (r["reason"] in ("missing P(V1|X=1)", "zero denominator")
                 or single.get(i, {}).get("status") == "undetermined under one model")}
    subsets = {
        "all": set(readings),
        "solvable (R)": {i for i, r in readings.items() if r["reading_R"] != "undecidable"},
        "text-undetermined (R)": {i for i, r in readings.items() if r["reading_R"] == "undecidable"},
        "certified undetermined": core,
        "contradictions (R)": {i for i, r in readings.items() if r["reading_R"] == "contradict"},
    }
    flip = {"yes": "no", "no": "yes"}
    literal = {i: (r["label"] if r["reading_E"] == "agree" else flip[r["label"]])
               for i, r in readings.items() if r["reading_E"] in ("agree", "contradict")}
    out = {"subsets": {k: len(v) for k, v in subsets.items()}, "models": {}}
    for f in sorted((ROOT / "results" / "model_eval").glob("*.jsonl")):
        answers = {x["id"]: x for x in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
        model = f.stem.replace("__", "/")
        row = {"answered": len(answers), "cost_usd": round(sum(a.get("cost") or 0 for a in answers.values()), 4)}
        for name, ids in subsets.items():
            got = [i for i in ids if i in answers]
            k = sum(answers[i]["answer"] == readings[i]["label"] for i in got)
            row[name] = {"tasks": len(got), "correct": k, "invalid": sum(answers[i]["answer"] == "invalid" for i in got),
                         "accuracy": round(k / len(got), 4) if got else None, "wilson95": wilson(k, len(got))}
        und = [i for i in subsets["text-undetermined (R)"] if i in answers and i in literal]
        same = sum(answers[i]["answer"] == literal[i] for i in und)
        row["undetermined: equals the literal answer"] = {"tasks": len(und), "equal": same,
                                                         "share": round(same / len(und), 4) if und else None}
        out["models"][model] = row
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
