"""Premise-level error injection: does the prompt matching catch a corrupted prompt?

    python -B scripts/perturb_check.py [--out results/perturb_check.json]

For every task that matches its prompt (match_check.py property ok), three corrupted copies of the prompt are made, each
with one change, and the matching is rerun against the unchanged structured fields:
  drop_probability  one sentence that states a probability is deleted;
  reverse_edge      one edge sentence "A has a direct effect on B." with a single effect becomes "B has a direct effect on A.";
  change_number     one printed percentage is changed by +7 points (or -7 above 90%).
The sentence or number is chosen by a seeded generator (seed 1), the same for every run. A corruption is caught when the
parser refuses the task or the premises read back from the text differ from those of the fields; it is missed when the
matching still passes. Tasks without a sentence of the kind are counted as not applicable. The refusal reason is
reported: the graph description (context, edge and latent sentences) must equal the text of one generator model, and
every probability sentence must match a premise by value and variables.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cladder" / "full_v1.5_default.csv"
META = ROOT / "data" / "raw" / "cladder_meta"
SENT = re.compile(r"(?<=[.?!])\s+")
EDGE = re.compile(r"^(.+?) has a direct effect on ([^,]+?)\.$")
PERCENT = re.compile(r"(\d+)%")


def sentences(prompt: str) -> list[str]:
    return [s for s in SENT.split(prompt.strip()) if s]


def drop_probability(prompt: str, rng: random.Random) -> str | None:
    parts = sentences(prompt)
    idx = [i for i, s in enumerate(parts[:-1]) if PERCENT.search(s)]
    if not idx:
        return None
    i = rng.choice(idx)
    return " ".join(parts[:i] + parts[i + 1:])


def reverse_edge(prompt: str, rng: random.Random) -> str | None:
    parts = sentences(prompt)
    idx = [i for i, s in enumerate(parts[:-1]) if EDGE.match(s) and " and " not in EDGE.match(s).group(2)
           and not s.startswith("Imagine")]
    if not idx:
        return None
    i = rng.choice(idx)
    a, b = EDGE.match(parts[i]).groups()
    parts[i] = f"{b[0].upper()}{b[1:]} has a direct effect on {a[0].lower()}{a[1:]}."
    return " ".join(parts)


def change_number(prompt: str, rng: random.Random) -> str | None:
    found = list(PERCENT.finditer(prompt))
    if not found:
        return None
    m = rng.choice(found)
    v = int(m.group(1))
    new = v + 7 if v <= 90 else v - 7
    return prompt[:m.start()] + f"{new}%" + prompt[m.end():]


KINDS = {"drop_probability": drop_probability, "reverse_edge": reverse_edge, "change_number": change_number}


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from cladder_audit.cladder import record_to_problem
    from cladder_audit.parsers import cladder as parser
    from cladder_audit.parsers.cladder_generator import Generator
    from cladder_audit.solver import solve
    from cladder_audit.transfer import MALFORMED, FreshNames, Omega, TransferError, transfer, untransfer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "perturb_check.json")
    ap.add_argument("--match", type=Path, default=ROOT / "results" / "match_check.json")
    args = ap.parse_args()
    ok_ids = {k.split(":")[1] for k, v in json.loads(args.match.read_text(encoding="utf-8"))["items"].items()
              if v.get("property")}
    generator = Generator.load(META)
    csv.field_size_limit(10**8)
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["id"] in ok_ids]
    rng = random.Random(1)
    names, omega = FreshNames(1, set()), Omega(seed=1, k=None)
    summary, missed = Counter(), {k: [] for k in KINDS}
    for row in rows:
        forms = generator.task_forms(row) if row["query_type"] == "det-counterfactual" else None
        expected = record_to_problem(row, forms)
        for kind, corrupt in KINDS.items():
            prompt = corrupt(row["prompt"], rng)
            if prompt is None:
                summary[f"{kind} not applicable"] += 1
                continue
            bad = {**row, "prompt": prompt}
            try:
                item = parser.parse(bad, generator)
                premises, question = untransfer(transfer(item, omega, names=names))
                same = (premises.ids() == expected[0].ids() and premises.latent == expected[0].latent
                        and all(premises.get(p.premise_id) == p.value for p in expected[0].items if p.kind != "equation")
                        and solve(premises, question) == solve(*expected))
                result = "missed" if same else "caught (premises differ)"
            except TransferError as exc:
                text = str(exc).replace(f"cladder:{row['id']}: ", "")
                if "generator models" in text:  # the edge and latent sentences must equal one generator model's text
                    reason = "graph description matches no generator model"
                else:
                    reason = re.sub(r" sentence .* matches ", " sentence matches ", text.split(": ")[0])
                result = f"caught (refused: {reason})"
            except MALFORMED as exc:  # a crash is reported apart: it is not a designed refusal
                result = f"error {type(exc).__name__}"
            summary[f"{kind} {result}"] += 1
            if result == "missed":
                missed[kind].append(row["id"])
    out = {"tasks": len(rows), "summary": dict(sorted(summary.items())),
           "missed": {k: sorted(v, key=int) for k, v in missed.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
