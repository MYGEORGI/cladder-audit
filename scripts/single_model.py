"""Single-model check of the rounding band: does one causal Bayesian network realize both answers?

    python -B scripts/single_model.py [--out results/single_model.json]   # needs scipy (optimizer only)

Reading R puts each printed probability in its own interval [p - 0.005, p + 0.005]. A vertex of that box need not come
from one probability model, so "the text does not determine the answer" may be an overcount. For every task of the
rounding band (reason "rounding band" or "denominator may vanish" in label_readings.csv) this script searches the
parameters of one binary Bayesian network on the task's graph (all nodes, unobserved ones included; one parameter
P(v=1 | parents) per node and parent configuration) whose implied printed probabilities all lie in their intervals and
whose recipe value gives the answer "yes", and another such network giving "no".

The search is numeric (SLSQP from scipy, seeded restarts); its result is then certified in exact arithmetic: the
parameters are rounded to fractions, the printed probabilities and the recipe value are recomputed as fractions, and a
side counts only if every printed probability is inside its closed interval and `decide` gives that answer.

Statuses: "one answer on the whole box" - the closed value interval touches the threshold, yet every vertex (hence every
point, `decide` being monotone in the value) gives the same answer, so reading R's interval test overcounts the task;
"threshold reached only on the box boundary" - the second answer needs the value exactly on the threshold at a box
corner (a printed probability exactly half a unit from its rounded value); "undetermined under one model" - both
answers certified, a lower bound on such tasks; "not certified" - the search found one answer only (proves nothing).
"""

import argparse
import csv
import itertools
import json
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cladder" / "full_v1.5_default.csv"
META = ROOT / "data" / "raw" / "cladder_meta"
EPS = Fraction(1, 200)


class Model:
    """Joint distribution of a binary Bayesian network on `nodes` with `parents`."""

    def __init__(self, nodes, parents):
        self.nodes, self.parents = nodes, parents
        self.slots = [(v, cfg) for v in nodes for cfg in itertools.product((0, 1), repeat=len(parents[v]))]

    def joint(self, theta, one=1.0):
        table = dict(zip(self.slots, theta, strict=True))
        cells = {}
        for values in itertools.product((0, 1), repeat=len(self.nodes)):
            a = dict(zip(self.nodes, values, strict=True))
            prob = one
            for v in self.nodes:
                q = table[(v, tuple(a[u] for u in self.parents[v]))]
                prob = prob * (q if a[v] == 1 else one - q)
            cells[values] = prob
        return cells

    def prob(self, cells, event, given=None, zero=0.0):
        given = given or {}
        idx = {v: i for i, v in enumerate(self.nodes)}

        def mass(cond):
            return sum((p for vals, p in cells.items() if all(vals[idx[k]] == x for k, x in cond.items())), zero)

        num, den = mass({**event, **given}), mass(given) if given else None
        return num if den is None else num / den


def parse_id(pid):
    """'p:Y=1|V1=0,X=1' -> ({'Y': 1}, {'V1': 0, 'X': 1}); 'j:X=1,Y=1' -> ({'X': 1, 'Y': 1}, {})."""
    kind, body = pid.split(":", 1)
    head, _, given = body.partition("|")

    def atoms(s):
        return {a.split("=")[0]: int(a.split("=")[1]) for a in s.split(",") if a}

    return atoms(head), atoms(given)


class ModelLookup:
    """The recipe's view of a model: P(var = val | given) and joints computed from the model's joint table."""

    def __init__(self, model, cells, zero):
        self.model, self.cells, self.zero = model, cells, zero

    def __call__(self, var, val, **given):
        return self.model.prob(self.cells, {var: val}, given, self.zero)

    def joint(self, **values):
        return self.model.prob(self.cells, values, None, self.zero)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from scipy.optimize import least_squares, minimize

    from cladder_audit.cladder import GRAPHS, record_to_problem
    from cladder_audit.parsers.cladder_generator import Generator
    from cladder_audit.premises import Premise, Premises
    from cladder_audit.solver import RECIPES, _evaluate, _IllFormed, _Missing, decide, graph_of, threshold

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "single_model.json")
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="first N band tasks only (a trial run)")
    args = ap.parse_args()
    readings = {r["id"]: r for r in csv.DictReader((ROOT / "results" / "label_readings.csv").open(encoding="utf-8"))}
    band = {i for i, r in readings.items() if r["reason"] in ("rounding band", "denominator may vanish")}
    generator = Generator.load(META)
    csv.field_size_limit(10**8)
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["id"] in band][: args.limit]
    report, counts = {}, Counter()
    for row in rows:
        forms = generator.task_forms(row) if row["query_type"] == "det-counterfactual" else None
        premises, question = record_to_problem(row, forms)
        graph = graph_of(premises) or row["graph_id"]
        name, recipe = RECIPES[(question.query_type, graph)]
        edges = GRAPHS[row["graph_id"]]
        nodes = sorted({n for e in edges for n in e} | {"X", "Y"}, key=lambda v: (v not in ("X", "Y"), v))
        parents = {v: sorted(a for a, b in edges if b == v) for v in nodes}
        model = Model(nodes, parents)
        printed = [(p.premise_id, p.value) for p in premises.items if p.kind in ("probability", "joint")]
        targets = [parse_id(pid) for pid, _ in printed]
        box = [(max(v - EPS, Fraction(0)), min(v + EPS, Fraction(1))) for _, v in printed]  # clipped as solver._band
        lo = np.array([float(a) for a, _ in box])
        hi = np.array([float(b) for _, b in box])

        def implied(theta, one=1.0, zero=0.0):
            cells = model.joint(theta, one)
            return cells, [model.prob(cells, ev, gv, zero) for ev, gv in targets]

        def value(theta, one=1.0, zero=0.0):
            cells, _ = implied(theta, one, zero)
            return recipe(ModelLookup(model, cells, zero))

        def certify(theta):
            q = [Fraction(float(t)).limit_denominator(10**6) for t in theta]
            q = [min(max(x, Fraction(1, 10**6)), 1 - Fraction(1, 10**6)) for x in q]
            try:
                cells, vals = implied(q, Fraction(1), Fraction(0))
                if not all(a <= x <= b for (a, b), x in zip(box, vals, strict=True)):
                    return None
                return decide(question.query_type, question.polarity, recipe(ModelLookup(model, cells, Fraction(0))))
            except ZeroDivisionError:
                return None

        def at(values):  # the recipe on printed values (the reading-R function itself), exact
            items = [Premise(p.premise_id, p.kind, values[p.premise_id]) if p.premise_id in values else p
                     for p in premises.items]
            try:
                return _evaluate(Premises.of(items, latent=premises.latent), name, recipe)[0]
            except (ZeroDivisionError, _Missing, _IllFormed):
                return None

        center = {pid: v for pid, v in printed}
        t = threshold(question.query_type)
        extreme = {}  # answer -> (distance from the threshold, vertex) of the most extreme vertex on that side
        for signs in itertools.product((-1, 1), repeat=len(printed)):
            vertex = {pid: (a if s < 0 else b) for (pid, _), (a, b), s in zip(printed, box, signs, strict=True)}
            f = at(vertex)
            if f is None:
                continue
            side = decide(question.query_type, question.polarity, f)
            if side not in extreme or abs(f - t) > extreme[side][0]:
                extreme[side] = (abs(f - t), vertex)
        touch = sorted(s for s, (dist, _) in extreme.items() if dist == 0)  # a side reached only on the threshold
        rng = random.Random(int(row["id"]))
        found = {}
        m = len(model.slots)

        def fit(target):  # parameters whose implied printed values are as close to `target` as possible
            goal = np.array([float(target[pid]) for pid, _ in printed])
            for start in [np.full(m, 0.5)] + [np.array([rng.uniform(0.05, 0.95) for _ in range(m)]) for _ in range(4)]:
                try:
                    res = least_squares(lambda th: np.array(implied(th)[1]) - goal, start, bounds=(1e-6, 1 - 1e-6),
                                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
                except (ZeroDivisionError, ValueError, FloatingPointError):
                    continue
                yield res.x

        for side, (dist, vertex) in sorted(extreme.items()):
            if dist == 0:
                continue  # an equality at a box corner: left to the boundary count
            for lam in (Fraction(1), Fraction(19, 20), Fraction(4, 5), Fraction(3, 5), Fraction(2, 5), Fraction(1, 5)):
                target = {pid: center[pid] + lam * (vertex[pid] - center[pid]) for pid in center}
                f = at(target)
                if f is None or decide(question.query_type, question.polarity, f) != side:
                    continue
                for theta in fit(target):
                    if certify(theta) == side:
                        found[side] = [str(Fraction(float(x)).limit_denominator(10**6)) for x in theta]
                        break
                if side in found:
                    break
        cons = [{"type": "ineq", "fun": lambda th: np.array(implied(th)[1]) - lo},
                {"type": "ineq", "fun": lambda th: hi - np.array(implied(th)[1])}]
        for sign in (1.0, -1.0):  # fallback: push the recipe value up, then down
            if len(found) == 2:
                break
            for _ in range(args.restarts):
                x0 = np.array([rng.uniform(0.02, 0.98) for _ in range(m)])
                try:
                    res = minimize(lambda th: -sign * float(value(th)), x0, method="SLSQP", constraints=cons,
                                   bounds=[(1e-6, 1 - 1e-6)] * m, options={"maxiter": 300, "ftol": 1e-12})
                except (ZeroDivisionError, ValueError, FloatingPointError):
                    continue
                answer = certify(res.x)
                if answer is not None and answer not in found:
                    found[answer] = [str(Fraction(float(x)).limit_denominator(10**6)) for x in res.x]
                if len(found) == 2:
                    break
        if len(extreme) == 1:  # the closed interval touches the threshold, but every point of the box gives one answer
            status = "one answer on the whole box"
        elif len(found) == 2:
            status = "undetermined under one model"
        elif touch:
            status = "threshold reached only on the box boundary"
        else:
            status = "not certified"
        counts[status] += 1
        counts[f"{question.query_type} {status}"] += 1
        report[row["id"]] = {"query_type": question.query_type, "graph": row["graph_id"], "status": status,
                             "witnesses": {a: dict(zip([f"{v}|{''.join(map(str, c))}" for v, c in model.slots], th,
                                                       strict=True)) for a, th in sorted(found.items())}}
    out = {"tasks": len(rows), "summary": dict(sorted(counts.items())), "items": dict(sorted(report.items(),
                                                                                          key=lambda kv: int(kv[0])))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
