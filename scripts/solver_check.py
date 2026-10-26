"""Check the reference solver against CLadder v1.5.

    python scripts/solver_check.py [--data <csv>] [--out <json>]

Every task lands in one category: agree; precision_conflict (the band holds the decision threshold); source_arithmetic
(the generator's printed result differs from the recipe on the same probabilities: the evidence is written);
source_missing_term and degenerate_premise (the solver cannot compute); ill_formed; unexplained. The label is read
here, outside the solver, only to compare.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

from cladder_audit.cladder import SolverInputError, record_to_problem
from cladder_audit.solver import solve

DATA = STAGE / "data" / "raw" / "cladder" / "full_v1.5_default.csv"
OUT = DATA.with_name("solver_check.json")
META = STAGE / "data" / "raw" / "cladder_meta"
_PRINTED = re.compile(r"=\s*(-?\d+\.\d+)\s*$")
_OPS = {}  # filled below: ast operator type -> function


def _init_ops():
    import ast
    import operator

    _OPS.update({ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv})


_init_ops()


def _printed_line(reasoning: str) -> str | None:
    """The generator's arithmetic line: the last line ending in '= <number>' before its comparison."""
    lines = [line.strip() for line in reasoning.split("\n") if line.strip()]
    return next((line for line in reversed(lines[:-1]) if _PRINTED.search(line)), None)


def _arithmetic(expression: str) -> Fraction | None:
    """Value of the generator's own expression ('0.72*0.53 - 0.28*0.40'): numbers, + - * / and parentheses only."""
    import ast

    try:
        node = ast.parse(expression.replace("−", "-"), mode="eval").body
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("not arithmetic")

    try:
        return ev(node)
    except (ValueError, ZeroDivisionError):
        return None


_TOKEN = re.compile(r"\\sum_\{(\w+)=(\w)\}|P\(([^)]*)\)|\d+\.\d+|\d+|[-+*/()\[\]]|\s+|−")


def _formula_value(formula: str, premises) -> Fraction | None:
    """The generator's own symbolic formula (4th line of `reasoning`, e.g. '\\sum_{V1=v} P(V1=v)*[...]') evaluated on
    the stated probabilities, independently of the solver's recipes. A sum covers the rest of its bracket group.
    Returns (value, premise ids it read); (None, empty) when the formula is not in this grammar or a probability is
    missing."""
    from cladder_audit.premises import joint_id, prob_id

    pos = 0
    used: set[str] = set()
    spans = [(m.group(0), m) for m in _TOKEN.finditer(formula.replace(" ", ""))]
    if "".join(s for s, _ in spans) != formula.replace(" ", ""):
        return None, frozenset()

    def atoms(text: str, env: dict, bare: str | None) -> dict[str, int]:
        out = {}
        for a in filter(None, text.split(",")):
            var, _, val = a.partition("=")
            if not val and bare is None:
                raise ValueError(f"no value for {var}")  # 'P(Y|X)' is shorthand, not P(Y=1|X=1)
            out[var] = int(env.get(val, val or bare))
        return out

    def prob(body: str, env: dict) -> Fraction:
        head, _, given = body.partition("|")
        event, cond = atoms(head, env, "1"), atoms(given, env, None)  # 'P(Y | X=1)': a bare event means Y = 1
        if len(event) > 1:
            v = premises.get(joint_id(event))
            if v is None:
                raise KeyError(body)
            used.add(joint_id(event))
            return v
        (var, val), = event.items()
        for value_id, flip in ((prob_id(var, val, cond), False), (prob_id(var, 1 - val, cond), True)):
            value = premises.get(value_id)
            if value is not None:
                used.add(value_id)
                return 1 - value if flip else value
        raise KeyError(body)

    def expr(env, end):
        nonlocal pos
        total, sign = Fraction(0), 1
        while pos < len(spans) and spans[pos][0] not in end:
            text, m = spans[pos]
            if text in "+-−":
                sign = -1 if text in "-−" else 1
                pos += 1
                continue
            total += sign * term(env, end)
            sign = 1
        return total

    def term(env, end):
        nonlocal pos
        value = factor(env, end)
        while pos < len(spans) and spans[pos][0] in "*/":
            op = spans[pos][0]
            pos += 1
            other = factor(env, end)
            value = value * other if op == "*" else value / other
        return value

    def factor(env, end):
        nonlocal pos
        text, m = spans[pos]
        pos += 1
        if m.group(1):  # \sum_{V=v}: the rest of the group, for v in 0, 1
            start, acc = pos, Fraction(0)
            for v in (0, 1):
                pos = start
                acc += expr({**env, m.group(2): v}, end)
            return acc
        if m.group(3) is not None:
            return prob(m.group(3), env)
        if text in "([":
            close = ")" if text == "(" else "]"
            value = expr(env, {close})
            pos += 1
            return value
        return Fraction(text)

    try:
        value = expr({}, set())
    except (KeyError, ValueError, IndexError, ZeroDivisionError):
        return None, frozenset()
    return (value, frozenset(used)) if pos == len(spans) else (None, frozenset())


def _foreign_terms(line: str, premises, used: frozenset[str]) -> list[str]:
    """Stated probabilities the generator's arithmetic uses although the generator's own formula does not read them."""
    allowed = {premises.get(i) for i in used} | {1 - premises.get(i) for i in used}
    foreign = []
    for p in premises.items:
        if p.kind in ("probability", "joint") and p.premise_id not in used and p.value not in allowed:
            if re.search(rf"(?<![\d.]){float(p.value):.2f}(?!\d)", line.rsplit("=", 1)[0]):
                foreign.append(p.premise_id)
    return foreign


def _source_evidence(row: dict, value: Fraction, forms=None) -> dict | None:
    """Proof, independent of the solver, that the generator's arithmetic is wrong.
    When the generator's own formula can be read, it must equal the solver's value (else the solver is suspect), and
    the arithmetic must contradict that formula, its own expression, or use a term the formula does not read. When it
    cannot be read, only a self-contradicting expression is proof. Disagreement with the solver alone never is."""
    line = _printed_line(row["reasoning"])
    if line is None:
        return None
    printed = Fraction(_PRINTED.search(line).group(1))
    own = _arithmetic(line.rsplit("=", 1)[0])
    premises, _ = record_to_problem(row, forms)
    lines = row["reasoning"].split("\n")
    formula, used = _formula_value(lines[3], premises) if len(lines) > 3 else (None, frozenset())
    if formula is not None and formula != value:
        return None
    foreign = _foreign_terms(line, premises, used) if formula is not None else []
    self_contradicting = own is not None and abs(own - printed) > Fraction(1, 200)
    formula_contradicting = formula is not None and abs(formula - printed) > Fraction(1, 200)
    if not (self_contradicting or foreign or formula_contradicting):
        return None
    return {"recomputed": f"{float(value):.4f}", "generator": f"{float(printed):.2f}", "generator_line": line,
            "generator_expression_value": None if own is None else f"{float(own):.4f}", "foreign_terms": foreign,
            "generator_formula_value": None if formula is None else f"{float(formula):.4f}"}


def _case(row: dict, forms=None) -> dict:
    case = {"id": row["id"], "query_type": row["query_type"], "graph_id": row["graph_id"], "label": row["label"]}
    try:
        out = solve(*record_to_problem(row, forms))
    except SolverInputError as error:
        return {**case, "category": "unexplained", "reason": str(error)}
    case.update(answer=out.answer, status=out.status, reason=out.reason, recipe=out.recipe)
    # independent guard: where the generator's own formula is readable, the solver must reproduce
    # it before any category, so a recipe defect never passes as agreement, rounding or an incomplete source
    lines = (row.get("reasoning") or "").split("\n")
    formula, _ = _formula_value(lines[3], record_to_problem(row, forms)[0]) if len(lines) > 3 else (None, frozenset())
    if formula is not None and (out.status != "ANSWER" or out.value != formula):
        return {**case, "category": "unexplained", "generator_formula_value": f"{float(formula):.4f}",
                "recomputed": None if out.value is None else f"{float(out.value):.4f}"}
    if out.status == "NOT_COMPUTABLE":
        category = "degenerate_premise" if out.reason == "ZERO_DENOMINATOR" else "source_missing_term"
        return {**case, "category": category, "missing": list(out.missing)}
    if out.status == "ILL_FORMED":
        return {**case, "category": "ill_formed", "missing": list(out.missing)}
    if out.answer == row["label"]:
        return {**case, "category": "agree"}
    if out.precision_conflict:
        return {**case, "category": "precision_conflict",
                "interval": None if out.interval is None else [f"{float(v):.4f}" for v in out.interval]}
    evidence = None if out.value is None else _source_evidence(row, out.value, forms)  # a graph answer has no value
    if evidence is not None:
        return {**case, "category": "source_arithmetic", **evidence}
    return {**case, "category": "unexplained", "recomputed": None if out.value is None else f"{float(out.value):.4f}"}


def check(rows: list[dict], generator) -> dict:
    """`generator` (parsers.cladder_generator.Generator) gives the forms a det-counterfactual question is read with."""
    cases = sorted((_case(row, generator.task_forms(row) if row["query_type"] == "det-counterfactual" else None)
                    for row in rows), key=lambda c: int(c["id"]))
    counts = Counter(c["category"] for c in cases)
    pairs: dict[str, Counter] = {}
    for c in cases:
        pairs.setdefault(f"{c['query_type']}|{c['graph_id']}", Counter())[c["category"]] += 1
    return {"tasks": len(cases), "counts": dict(sorted(counts.items())),
            "pairs": {k: dict(sorted(v.items())) for k, v in sorted(pairs.items())},
            "cases": [c for c in cases if c["category"] != "agree"] + [c for c in cases if c["category"] == "agree"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--meta", type=Path, default=META, help="CLadder generator files (scripts/fetch_cladder.py)")
    args = parser.parse_args(argv)
    from cladder_audit.parsers.cladder_generator import Generator

    csv.field_size_limit(10**8)
    with args.data.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = {"data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(), **check(rows, Generator.load(args.meta))}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["counts"]))
    return 0 if report["counts"].get("unexplained", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
