"""Every rule-based repair the reader makes to the `reasoning` field of a CLadder v1.5 record, task by task.

    python -B scripts/repairs.py [--out results/repairs.csv]

Two generator typos are repaired; nothing else in a record is changed, and neither the prompt nor the label is ever
touched. `typo P(X=1=1)`: a probability line of the correlation recipe reads P(X=1=1); it is read as P(X=1).
`equation operand`: in a det-counterfactual structural equation the single operand of a variable with one parent is not
that parent; it is replaced by the parent from the graph. The column `prompt_match` gives the result of
match_check.py for the task (the repaired premises read back from the prompt text): `ok`, or the refusal category.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cladder" / "full_v1.5_default.csv"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from cladder_audit.cladder import _EQUATION, GRAPHS, premises_from_fields, repair_equation

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "repairs.csv")
    ap.add_argument("--match", type=Path, default=ROOT / "results" / "match_check.json")
    args = ap.parse_args()
    match = json.loads(args.match.read_text(encoding="utf-8"))["items"]
    csv.field_size_limit(10**8)
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out, counts = [], Counter()
    for row in rows:
        reasoning = row.get("reasoning") or ""
        if reasoning.strip().startswith("nan"):
            reasoning = ""
        entry = match.get(f"cladder:{row['id']}", {})
        matched = entry.get("category") or ("ok" if entry.get("property") else "failed")
        _, notes = premises_from_fields(row["graph_id"], reasoning)
        for line in reasoning.split("\n"):
            if "typo P(X=1=1)" in notes and "=1=1" in line:
                out.append([row["id"], row["query_type"], row["graph_id"], "typo P(X=1=1)", line.strip(),
                            line.strip().replace("=1=1", "=1"), matched])
        parents: dict[str, set[str]] = {}
        for a, b in GRAPHS[row["graph_id"]]:
            parents.setdefault(b, set()).add(a)
        for line in reasoning.split("\n")[1:]:
            m = _EQUATION.match(line.strip())
            if m:
                fixed = repair_equation(m.group(1), m.group(2).strip(), parents)
                if fixed != m.group(2).strip():
                    out.append([row["id"], row["query_type"], row["graph_id"], "equation operand",
                                f"{m.group(1)} = {m.group(2).strip()}", f"{m.group(1)} = {fixed}", matched])
    tasks = {}
    for r in out:
        tasks.setdefault((r[0], r[3]), r[6])
    for (tid, kind), matched in tasks.items():
        counts[(kind, matched)] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["id", "query_type", "graph_id", "repair", "as_printed", "as_read", "prompt_match"])
        w.writerows(sorted(out, key=lambda r: (int(r[0]), r[3], r[4])))
    print(json.dumps({f"{k} | {m}": n for (k, m), n in sorted(counts.items())}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
