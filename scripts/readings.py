"""Label-level verdict of every CLadder v1.5 task under two readings of the printed numbers.

    python -B readings.py <artifact root> <out dir>

Reading E (as printed): the printed probabilities are exact. Reading R (rounded): each printed probability stands for
any value within +-0.005 (two decimals), so a task is decidable only if the value interval excludes the threshold.
Verdicts: agree, contradict (the text decides the other answer), undecidable (the text decides no answer). Each verdict
uses the text alone; the label is read only to compare. ETT on `confounding` lacks P(V1=1|X=1): its value is the mean
of the two stratum differences with an unknown weight in [0, 1], so the interval is taken over that weight as well.

Writes <out dir>/label_readings.csv (with the value range of the estimand under each reading: E_low..E_high and
R_low..R_high, empty where the task has no numeric target or the range is unbounded), <out dir>/solvable_subset.txt
(ids whose reading R is not undecidable) and <out dir>/label_readings.json (counts and a planted-flip check).
"""

import csv
import importlib.util
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

stage, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(stage)]
spec = importlib.util.spec_from_file_location("solver_check", stage / "scripts" / "solver_check.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
from cladder_audit.cladder import record_to_problem
from cladder_audit.parsers.cladder_generator import Generator
from cladder_audit.solver import decide, solve

EPS = Fraction(1, 200)
csv.field_size_limit(10**8)
with (stage / "data" / "raw" / "cladder" / "full_v1.5_default.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
generator = Generator.load(stage / "data" / "raw" / "cladder_meta")


def verdict(answers, label):
    """`answers` is the set of answers the text allows."""
    if len(answers) != 1:
        return "undecidable"
    return "agree" if next(iter(answers)) == label else "contradict"


def ett_missing(row, question):
    """Answers allowed by the text for ETT on confounding: weight w = P(V1=1|X=1) in [0, 1] is not stated."""
    p = {(int(a), int(b)): Fraction(c)
         for a, b, c in re.findall(r"P\(Y=1 \| V1=(\d), X=(\d)\) = ([0-9.]+)", row["reasoning"])}
    d = [p[(v, 1)] - p[(v, 0)] for v in (0, 1)]
    exact = {decide("ett", question.polarity, d[v]) for v in (0, 1)}  # the extremes w = 0 and w = 1
    rounded = {decide("ett", question.polarity, d[v] + s * 2 * EPS) for v in (0, 1) for s in (-1, 0, 1)}
    return exact, rounded, (min(d), max(d)), (min(d) - 2 * EPS, max(d) + 2 * EPS)


def fmt(rng):
    return ["", ""] if rng is None else [f"{float(x):.6f}" for x in rng]


records, counts, planted = [], Counter(), Counter()
for row in rows:
    forms = generator.task_forms(row) if row["query_type"] == "det-counterfactual" else None
    category = sc._case(row, forms)["category"]
    premises, question = record_to_problem(row, forms)
    o = solve(premises, question)
    if category == "source_missing_term":
        exact, rounded, e_rng, r_rng = ett_missing(row, question)
        reason = "missing P(V1|X=1)"
    elif category == "degenerate_premise":
        exact = rounded = set()
        e_rng = r_rng = None
        reason = "zero denominator"
    elif o.value is not None and o.interval is None and row["query_type"] != "det-counterfactual":
        # solver._band gives no interval when a denominator may vanish inside the box: the ratio then takes both signs
        exact, rounded = {o.answer}, {"yes", "no"}
        e_rng, r_rng = (o.value, o.value), None
        reason = "denominator may vanish"
    else:
        assert o.status == "ANSWER", (row["id"], o.status)  # only the two categories above leave the solver without an answer
        exact = {o.answer}
        rounded = {"yes", "no"} if o.precision_conflict else {o.answer}
        reason = "rounding band" if o.precision_conflict else ""
        e_rng = (o.value, o.value) if o.value is not None else None
        r_rng = o.interval
    e, r = verdict(exact, row["label"]), verdict(rounded, row["label"])
    counts[("E", e)] += 1
    counts[("R", r)] += 1
    counts[("E|R", f"{e}|{r}")] += 1
    records.append([row["id"], row["rung"], row["query_type"], row["graph_id"], row["label"], e, r, reason,
                    *fmt(e_rng), *fmt(r_rng)])
    # logic identity, not a test: the verdict ignores the label by construction (the real spike-in is band_and_spikein.py)
    flipped = "no" if row["label"] == "yes" else "yes"
    planted[("E", e, verdict(exact, flipped))] += 1
    planted[("R", r, verdict(rounded, flipped))] += 1

assert counts[("E", "agree")] + counts[("E", "contradict")] + counts[("E", "undecidable")] == len(rows) == 10112
out.mkdir(parents=True, exist_ok=True)
with (out / "label_readings.csv").open("w", encoding="utf-8", newline="") as handle:
    w = csv.writer(handle, lineterminator="\n")
    w.writerow(["id", "rung", "query_type", "graph_id", "label", "reading_E", "reading_R", "reason",
                "E_low", "E_high", "R_low", "R_high"])
    w.writerows(sorted(records, key=lambda r: int(r[0])))
solvable = sorted(int(r[0]) for r in records if r[6] != "undecidable")
(out / "solvable_subset.txt").write_text("".join(f"{i}\n" for i in solvable), encoding="utf-8", newline="\n")
by_query = Counter((r[2], r[6]) for r in records)
report = {
    "counts": {" ".join(k): v for k, v in sorted(counts.items())},
    "reading_R_by_query_type": {" ".join(k): v for k, v in sorted(by_query.items())},
    "planted_flip": {" ".join(k): v for k, v in sorted(planted.items())},
}
(out / "label_readings.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(report, indent=1))
