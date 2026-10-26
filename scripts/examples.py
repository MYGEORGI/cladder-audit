"""One worked example per defect class for the paper, with the witnesses of undecidability.

    python -B examples.py <artifact root> <data dir>

Picks the lowest task id of each class from label_readings.csv and cladder_v15_defects.csv, prints the prompt, the
probabilities, the label, the solver's value and band, and for ETT a pair of P(X|V1) values that the text allows and that
give opposite answers.
"""

import csv
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

stage, data = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(stage)]
from cladder_audit.cladder import record_to_problem
from cladder_audit.parsers.cladder_generator import Generator
from cladder_audit.solver import decide, solve

csv.field_size_limit(10**8)
rows = {r["id"]: r for r in csv.DictReader((stage / "data/raw/cladder/full_v1.5_default.csv").open(encoding="utf-8", newline=""))}
readings = list(csv.DictReader((data / "label_readings.csv").open(encoding="utf-8", newline="")))
defects = list(csv.DictReader((data / "cladder_v15_defects.csv").open(encoding="utf-8", newline="")))
generator = Generator.load(stage / "data/raw/cladder_meta")


def first(items, test):
    return min((r for r in items if test(r)), key=lambda r: int(r["id"]))


picks = {
    "band, agrees as printed (E agree, R undecidable)": first(readings, lambda r: r["reading_E"] == "agree" and r["reason"] == "rounding band"),
    "band, contradicts as printed (E contradict, R undecidable)": first(readings, lambda r: r["reading_E"] == "contradict" and r["reason"] == "rounding band"),
    "contradicts the text (R contradict)": first(readings, lambda r: r["reading_R"] == "contradict"),
    "ETT missing term, undecidable": first(readings, lambda r: r["reason"].startswith("missing") and r["reading_E"] == "undecidable"),
    "equation mismatch": first(defects, lambda r: r["category"] == "source_equation_mismatch"),
    "alias collision": first(defects, lambda r: r["category"] == "source_alias_collision"),
}
for name, pick in picks.items():
    row = rows[pick["id"]]
    forms = generator.task_forms(row) if row["query_type"] == "det-counterfactual" else None
    print(f"\n===== {name}: task {row['id']} ({row['query_type']}, {row['graph_id']}, story {row['story_id']}), label {row['label']}")
    print(row["prompt"])
    print("--- reasoning:", row["reasoning"].replace("\n", " | "))
    if pick.get("detail"):
        print("--- defect detail:", pick["detail"])
    try:
        o = solve(*record_to_problem(row, forms))
        print("--- solver:", o.status, o.answer, None if o.value is None else float(o.value),
              None if o.interval is None else [float(v) for v in o.interval], "band" if o.precision_conflict else "")
    except Exception as error:  # the example list must still print the other classes
        print("--- solver error:", error)
    if name.startswith("ETT"):
        p = {(int(a), int(b)): Fraction(c) for a, b, c in re.findall(r"P\(Y=1 \| V1=(\d), X=(\d)\) = ([0-9.]+)", row["reasoning"])}
        pv = Fraction(re.search(r"P\(V1=1\) = ([0-9.]+)", row["reasoning"])[1])
        d = [p[(v, 1)] - p[(v, 0)] for v in (0, 1)]
        _, q = record_to_problem(row)
        for a, b in ((Fraction(1, 10), Fraction(9, 10)), (Fraction(9, 10), Fraction(1, 10))):  # P(X=1|V1=1), P(X=1|V1=0)
            w = pv * a / (pv * a + (1 - pv) * b)  # P(V1=1|X=1)
            ett = (1 - w) * d[0] + w * d[1]
            print(f"--- witness P(X=1|V1=1)={a}, P(X=1|V1=0)={b}: P(V1=1|X=1)={float(w):.4f}, ETT={float(ett):.4f}, "
                  f"answer {decide('ett', q.polarity, ett)}")
