"""Matching of every CLadder v1.5 prompt against the premises the solver reads from the structured fields.

    python scripts/match_check.py [--out results/match_check.json]

Each task is parsed into prompt sentences, each paired with the premise it states (by value and by the generator's
verbal forms of the variables); a prompt sentence with a probability or an equation that matches no premise, or a
premise absent from the prompt, refuses the task. The parsed task is then renamed with fresh variable names and read
back: the property holds when the premises, their values and the solver's outcome come back unchanged. The report
lists every task, its property or its refusal category, and the source_alias_collision flag (one phrase for both values
of a variable). Two runs on the same input are byte-identical.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cladder" / "full_v1.5_default.csv"
META = ROOT / "data" / "raw" / "cladder_meta"


def category(exc: Exception, tid: str, transfer_error: type) -> str:
    text = str(exc)
    if isinstance(exc, transfer_error) and text.startswith(f"{tid}: "):
        return text[len(tid) + 2:].split(": ")[0]
    return type(exc).__name__


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from cladder_audit.cladder import record_to_problem
    from cladder_audit.parsers import cladder as parser
    from cladder_audit.parsers.cladder_generator import Generator
    from cladder_audit.solver import solve
    from cladder_audit.transfer import MALFORMED, FreshNames, Omega, TransferError, transfer, untransfer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "match_check.json")
    args = ap.parse_args()
    generator = Generator.load(META)
    csv.field_size_limit(10**8)
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report, items = {}, []
    for row in rows:
        tid = f"cladder:{row['id']}"
        try:
            items.append((parser.parse(row, generator), row))
        except MALFORMED as exc:
            report[tid] = {"error": str(exc), "category": category(exc, tid, TransferError)}
    words = {w.lower() for item, _ in items for text in [item.question, *(p.text for p in item.premises)]
             for w in text.replace(".", " ").split()}
    names, omega = FreshNames(1, words), Omega(seed=1, k=None)
    for item, row in items:
        expected = record_to_problem(row, item.forms)
        try:
            premises, question = untransfer(transfer(item, omega, names=names))
            ok = (premises.ids() == expected[0].ids() and premises.latent == expected[0].latent
                  and all(premises.get(p.premise_id) == p.value for p in expected[0].items if p.kind != "equation")
                  and solve(premises, question) == solve(*expected))
        except TransferError:
            ok = False
        flags = {"flags": ["source_alias_collision"]} if item.forms.collisions else {}
        report[item.item_id] = {"property": ok, **flags}
    summary = Counter()
    for entry in report.values():
        key = "error " + entry["category"] if "error" in entry else f"property {'ok' if entry['property'] else 'failed'}"
        summary[key] += 1
        summary.update(entry.get("flags", []))
    out = {"input_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(), "items": dict(sorted(report.items())),
           "summary": dict(sorted(summary.items()))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
