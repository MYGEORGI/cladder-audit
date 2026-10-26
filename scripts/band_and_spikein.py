"""Two checks of the audit itself, on CLadder v1.5 (paper on its ground truth).

    python -B band_and_spikein.py <artifact root> <out json>

1. Band census: for every task, does the value interval under +-0.005 per printed probability contain the decision
   threshold? solver_check reports this only when the answer differs from the label, so agreeing tasks inside the band
   are counted here.
2. Spike-in: flip the label of every task that agrees, and recategorise it. Each explained category needs evidence that
   does not come from the label (the band, a printed generator contradiction, a named missing term), so a planted error
   must land in `unexplained` unless that evidence exists for the task.
"""

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

stage, out_path = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(stage)]
spec = importlib.util.spec_from_file_location("solver_check", stage / "scripts" / "solver_check.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
from cladder_audit.cladder import record_to_problem
from cladder_audit.parsers.cladder_generator import Generator
from cladder_audit.solver import solve

csv.field_size_limit(10**8)
with (stage / "data" / "raw" / "cladder" / "full_v1.5_default.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
generator = Generator.load(stage / "data" / "raw" / "cladder_meta")


def forms(row):
    return generator.task_forms(row) if row["query_type"] == "det-counterfactual" else None


def evidence(case):
    """Which test of solver_check._source_evidence fired: the printed expression against its own result, a stated
    probability the recipe does not read, or the generator's formula line against the printed result."""
    kinds = []
    gen, own, formula = (case.get(k) for k in ("generator", "generator_expression_value", "generator_formula_value"))
    if own is not None and abs(float(own) - float(gen)) > 0.005:
        kinds.append("self_contradicting")
    if case.get("foreign_terms"):
        kinds.append("foreign_terms")
    if formula is not None and abs(float(formula) - float(gen)) > 0.005:
        kinds.append("formula_contradicting")
    return "+".join(kinds) or "none"


band, spike, by_query, real_ev, spike_ev = Counter(), Counter(), Counter(), Counter(), Counter()
for row in rows:
    f = forms(row)
    case = sc._case(row, f)
    out = solve(*record_to_problem(row, f))
    in_band = bool(getattr(out, "precision_conflict", False))
    band[(case["category"], "band" if in_band else "no band")] += 1
    if in_band:
        by_query[row["query_type"]] += 1
    if case["category"] == "source_arithmetic":
        real_ev[(row["query_type"], evidence(case))] += 1
    if case["category"] == "agree":
        flipped = dict(row, label="no" if row["label"] == "yes" else "yes")
        planted = sc._case(flipped, f)
        spike[planted["category"]] += 1
        if planted["category"] == "source_arithmetic":
            spike_ev[(row["query_type"], evidence(planted))] += 1

assert sum(band.values()) == len(rows) == 10112
report = {
    "band_census": {" | ".join(k): v for k, v in sorted(band.items())},
    "band_by_query_type": dict(sorted(by_query.items())),
    "spike_in_flipped_agree": dict(sorted(spike.items())),
    "source_arithmetic_evidence_real": {" | ".join(k): v for k, v in sorted(real_ev.items())},
    "source_arithmetic_evidence_planted": {" | ".join(k): v for k, v in sorted(spike_ev.items())},
}
out_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(report, indent=1))
