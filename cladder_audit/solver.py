"""Reference solver with recomputation.

The answer of a CLadder question is the benchmark recipe of its query type and graph, evaluated on exact fractions
and compared with the decision threshold of its type. The label is never an input.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable

from .premises import Outcome, Premises, Question, joint_id, prob_id

_UP = {"increase", "more", "larger", "positive"}
_DOWN = {"decrease", "less", "smaller", "negative"}


class _Missing(Exception):
    """A recipe term absent from the premises; args[0] is its premise id (or a tuple of ids)."""


class _IllFormed(Exception):
    """Premises that define no consistent model; args = (reason, premise id)."""


class _Lookup:
    """P(var = val | given) from the premises: the stated value or one minus its complement (CLadder states val = 1)."""

    def __init__(self, premises: Premises, fill: dict[str, Fraction] | None = None):
        self.premises = premises
        self.fill = fill or {}  # trial values of absent terms, keyed by their val = 1 id
        self.used: set[str] = set()  # premise ids the recipe read
        self.derived = False  # a conditional was derived by division from a joint and a marginal

    def _stated(self, var: str, val: int, given: dict) -> Fraction | None:
        for value_id, flip in ((prob_id(var, val, given), False), (prob_id(var, 1 - val, given), True)):
            value = self.premises.get(value_id)
            if value is not None:
                self.used.add(value_id)
                return 1 - value if flip else value
        return None

    def __call__(self, var: str, val: int, **given: int) -> Fraction:
        value = self._stated(var, val, given)
        if value is not None:
            return value
        if len(given) == 1:  # P(var=1 | c=v) = P(var=1, c=v) / P(c=v) when the joint and the marginal are stated
            (c, v), = given.items()
            joint = self.premises.get(joint_id({var: 1, c: v}))
            marginal = self._stated(c, v, {})
            if joint is not None and marginal:
                if joint > marginal:  # no probability model has P(A, B) > P(B)
                    raise _IllFormed("INCOMPATIBLE_PROBABILITIES", joint_id({var: 1, c: v}), prob_id(c, 1))
                self.used.add(joint_id({var: 1, c: v}))
                self.derived = True  # a division: the vertex argument of the precision band no longer holds
                p1 = joint / marginal
                return p1 if val == 1 else 1 - p1
        absent = prob_id(var, 1, given)
        if absent in self.fill:
            return self.fill[absent] if val == 1 else 1 - self.fill[absent]
        raise _Missing(absent)

    def joint(self, **values: int) -> Fraction:
        value = self.premises.get(joint_id(values))
        if value is not None:
            self.used.add(joint_id(values))
            return value
        if len(values) == 2 and values.get("Y") == 1:  # P(Y=1, X=x) = P(Y=1 | X=x) P(X=x)
            (c, v), = ((k, x) for k, x in values.items() if k != "Y")
            conditional, marginal = self._stated("Y", 1, {c: v}), self._stated(c, v, {})
            if conditional is not None and marginal is not None:
                return conditional * marginal
        raise _Missing(joint_id(values))


_RATIONAL = {"iv_wald", "correlation"}  # recipes with a division: a missing term is never settled by vertices
_MAX_MISSING = 8  # at most 2**8 vertices; CLadder recipes have at most 9 terms


def _evaluate(premises: Premises, name: str, recipe: Callable) -> tuple[Fraction, tuple[str, ...]]:
    """Recipe value and the absent terms it did not need. A multilinear recipe is affine in each probability, so
    a value equal at every vertex of the absent terms' box [0, 1]^k holds for any value of them. Raises _Missing with
    all absent ids when the value depends on them (or the recipe divides)."""
    from itertools import product

    absent: list[str] = []
    while True:
        try:
            value = recipe(_Lookup(premises, {a: Fraction(0) for a in absent}))
            break
        except _Missing as missing:
            if name in _RATIONAL or len(absent) >= _MAX_MISSING:
                raise _Missing(tuple(sorted({*absent, missing.args[0]}))) from None
            absent.append(missing.args[0])
    if not absent:
        return value, ()
    values = {recipe(_Lookup(premises, dict(zip(absent, vertex, strict=True))))
              for vertex in product((Fraction(0), Fraction(1)), repeat=len(absent))}
    if len(values) > 1:
        raise _Missing(tuple(sorted(absent)))
    return value, tuple(sorted(absent))


def recipe_inputs(premises: Premises, question: Question) -> frozenset[str]:
    """Probability ids the benchmark recipe of a numeric question reads, including the absent terms it needed to try.
    Empty if the recipe cannot be computed."""
    key = (question.query_type, graph_of(premises))
    if key not in RECIPES:
        return frozenset()
    name, recipe = RECIPES[key]
    try:
        _, absent = _evaluate(premises, name, recipe)
        lookup = _Lookup(premises, {a: Fraction(0) for a in absent})
        recipe(lookup)
    except (_Missing, _IllFormed, ZeroDivisionError):
        return frozenset()
    return frozenset(lookup.used) | frozenset(absent)


_S = (0, 1)


def _no_confounding(p):
    return p("Y", 1, X=1) - p("Y", 1, X=0)


def _backdoor(p):
    return sum(p("V1", v) * (p("Y", 1, V1=v, X=1) - p("Y", 1, V1=v, X=0)) for v in _S)


def _ett_backdoor(p):
    return sum(p("V1", v, X=1) * (p("Y", 1, V1=v, X=1) - p("Y", 1, V1=v, X=0)) for v in _S)


def _ett_v3(p):
    return sum(p("V3", v, X=1) * (p("Y", 1, X=1, V3=v) - p("Y", 1, X=0, V3=v)) for v in _S)


def _frontdoor(p):
    return sum((p("V3", v, X=1) - p("V3", v, X=0)) * sum(p("Y", 1, X=h, V3=v) * p("X", h) for h in _S) for v in _S)


def _ett_frontdoor(p):
    return sum(p("Y", 1, X=1, V3=v) * (p("V3", v, X=1) - p("V3", v, X=0)) for v in _S)


def _iv_wald(p):
    return (p("Y", 1, V2=1) - p("Y", 1, V2=0)) / (p("X", 1, V2=1) - p("X", 1, V2=0))


def _nde_mediation(p):
    return sum(p("V2", v, X=0) * (p("Y", 1, X=1, V2=v) - p("Y", 1, X=0, V2=v)) for v in _S)


def _nie_mediation(p):
    return sum(p("Y", 1, X=0, V2=v) * (p("V2", v, X=1) - p("V2", v, X=0)) for v in _S)


def _nde_arrowhead(p):
    return sum((p("Y", 1, X=1, V3=v) - p("Y", 1, X=0, V3=v)) * p("V3", v, X=0, V2=k) * p("V2", k) for v in _S for k in _S)


def _nie_arrowhead(p):
    return sum(p("Y", 1, X=0, V3=v) * (p("V3", v, X=1, V2=k) - p("V3", v, X=0, V2=k)) * p("V2", k) for v in _S for k in _S)


def _marginal(p):
    return sum(p("Y", 1, X=x) * p("X", x) for x in _S)


def _correlation(p):
    px1 = p("X", 1)
    return p.joint(X=1, Y=1) / px1 - p.joint(X=0, Y=1) / (1 - px1)


def _exp_away(p):
    return p("Y", 1, X=1, V3=1) - sum(p("X", x) * p("Y", 1, X=x, V3=1) for x in _S)


_NO_CONF = ("chain", "mediation", "diamond", "arrowhead", "fork")

RECIPES: dict[tuple[str, str], tuple[str, Callable]] = {
    **{("ate", g): ("no_confounding", _no_confounding) for g in _NO_CONF},
    **{("ett", g): ("no_confounding", _no_confounding) for g in _NO_CONF},
    **{("nie", g): ("no_confounding", _no_confounding) for g in ("chain", "diamond")},
    **{("ate", g): ("backdoor", _backdoor) for g in ("confounding", "diamondcut")},
    ("ett", "confounding"): ("ett_backdoor", _ett_backdoor),
    ("ett", "diamondcut"): ("ett_v3", _ett_v3),
    ("ate", "frontdoor"): ("frontdoor", _frontdoor),
    ("nie", "frontdoor"): ("frontdoor", _frontdoor),
    ("ett", "frontdoor"): ("ett_frontdoor", _ett_frontdoor),
    ("ate", "IV"): ("iv_wald", _iv_wald),
    ("nde", "mediation"): ("nde_mediation", _nde_mediation),
    ("nie", "mediation"): ("nie_mediation", _nie_mediation),
    ("nde", "arrowhead"): ("nde_arrowhead", _nde_arrowhead),
    ("nie", "arrowhead"): ("nie_arrowhead", _nie_arrowhead),
    ("exp_away", "collision"): ("exp_away", _exp_away),
    **{("marginal", g): ("marginal", _marginal) for g in ("IV", "arrowhead", "chain", "collision", "confounding",
                                                          "diamond", "diamondcut", "fork", "frontdoor", "mediation")},
    **{("correlation", g): ("correlation", _correlation) for g in ("IV", "arrowhead", "chain", "confounding", "diamond",
                                                                  "diamondcut", "fork", "frontdoor", "mediation")},
}


def _digraph(premises: Premises):
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_edges_from(tuple(p.premise_id[len("edge:"):].split("->")) for p in premises.items if p.kind == "edge")
    return graph


def _collider_bias(premises: Premises, question: Question) -> str:
    """Conditioning on a collider says nothing about an effect; the answer is whether the graph has a path X -> Y."""
    import networkx as nx

    affects = nx.has_path(_digraph(premises), question.treatment, question.outcome)
    return "yes" if affects == (question.polarity == "affects") else "no"


def _backdoor_valid(premises: Premises, question: Question, adjust: tuple[str, ...]) -> bool:
    """Back-door criterion on the graph: no descendant of X, and d-separation of X and Y once X's out-edges go.
    The benchmark recipe ignores observability: CLadder calls adjusting for a latent confounder correct."""
    import networkx as nx

    graph, x, y = _digraph(premises), question.treatment, question.outcome
    if set(adjust) & nx.descendants(graph, x):
        return False
    cut = graph.copy()
    cut.remove_edges_from(list(graph.out_edges(x)))
    return nx.is_d_separator(cut, {x}, {y}, set(adjust))


def _backadj(premises: Premises, question: Question) -> str:
    """'Is Method 1 more correct than Method 2?': yes when the first set satisfies the criterion and the second does
    not, or both do and the first is smaller (CLadder prefers not adjusting when adjusting is unnecessary)."""
    first, second = question.methods
    if question.polarity == "method2":
        first, second = second, first
    ok1, ok2 = (_backdoor_valid(premises, question, m) for m in (first, second))
    return "yes" if ok1 and (not ok2 or len(first) < len(second)) else "no"


def _boolean(expression: str, values: dict[str, int]) -> int:
    """Evaluate 'not'/'and'/'or' over variables (precedence not > and > or)."""
    disjuncts = expression.split(" or ")
    result = 0
    for disjunct in disjuncts:
        conj = 1
        for literal in disjunct.split(" and "):
            words = literal.split()
            negations, name = len(words) - 1, words[-1]
            if any(w != "not" for w in words[:-1]):
                raise ValueError(f"unreadable equation term {literal!r}")
            conj &= values[name] ^ (negations % 2)
        result |= conj
    return result


def _counterfactual(premises: Premises, question: Question) -> int:
    """Boolean SCM: evidence fixes the variables without an equation, the action replaces X's equation."""
    equations = {p.premise_id[len("eq:"):]: p.value for p in premises.items if p.kind == "equation"}
    graph = _digraph(premises)
    for var, expression in equations.items():
        used = {w for w in expression.split() if w not in ("not", "and", "or")}
        if var not in graph or not used <= set(graph.predecessors(var)):  # e.g. 'V3 = X' where V3's parent is V1
            raise _IllFormed("EQUATION_GRAPH_MISMATCH", f"eq:{var}")
    fixed = dict(question.evidence) | dict(question.intervention)
    values: dict[str, int] = {}

    def value(var: str) -> int:
        if var in values:
            return values[var]
        if var in fixed and (var in dict(question.intervention) or var not in equations):
            values[var] = fixed[var]
        elif var in equations:
            names = [w for w in equations[var].split() if w not in ("not", "and", "or")]
            values[var] = _boolean(equations[var], {n: value(n) for n in names})
        else:
            raise _Missing(f"eq:{var}")
        return values[var]

    return value(question.outcome)


EPS = Fraction(1, 200)  # CLadder states probabilities with two decimals: each is exact to +-0.005


def _iv_denominator(p):
    return p("X", 1, V2=1) - p("X", 1, V2=0)


def _correlation_denominators(p):
    return p("X", 1), 1 - p("X", 1)


_DENOMINATORS = {"iv_wald": lambda p: (_iv_denominator(p),), "correlation": _correlation_denominators}


def _band(premises: Premises, name: str, recipe: Callable,
          absent: tuple[str, ...] = ()) -> tuple[Fraction, Fraction] | None:
    """Range of the recipe over the box of +-EPS around every stated probability and [0, 1] for the absent terms the
    value did not need. Multilinear and linear-fractional recipes reach their extremes at the vertices while
    no denominator crosses zero; None if one can."""
    from itertools import product

    from .premises import Premise

    probs = [p for p in premises.items if p.kind in ("probability", "joint")]
    others = tuple(p for p in premises.items if p.kind not in ("probability", "joint"))
    values, signs = [], set()
    for corner in product((-EPS, EPS), repeat=len(probs)):
        shifted = tuple(Premise(p.premise_id, p.kind, min(max(p.value + d, Fraction(0)), Fraction(1)))
                        for p, d in zip(probs, corner, strict=True))
        moved = Premises(shifted + others, premises.latent, premises.assumptions)
        try:
            signs.update((i, (d > 0) - (d < 0))
                         for i, d in enumerate(_DENOMINATORS.get(name, lambda p: ())(_Lookup(moved))))
            for vertex in product((Fraction(0), Fraction(1)), repeat=len(absent)):
                lookup = _Lookup(moved, dict(zip(absent, vertex, strict=True)))
                values.append(recipe(lookup))
                if lookup.derived:  # a derived ratio may peak inside the box: no vertex enclosure
                    return None
        except (ZeroDivisionError, _Missing, _IllFormed):  # the input domain changes inside the box: band unknown
            return None
    # a denominator that is zero at a vertex, or changes sign between vertices, is zero somewhere in the box
    if any(s == 0 for _, s in signs) or len(signs) > len({i for i, _ in signs}):
        return None
    return min(values), max(values)


def graph_of(premises: Premises) -> str | None:
    """The CLadder graph whose edge set the premises state, else None."""
    from .cladder import GRAPHS

    edges = {p.premise_id[len("edge:"):] for p in premises.items if p.kind == "edge"}
    for name, graph_edges in GRAPHS.items():
        if edges == {f"{a}->{b}" for a, b in graph_edges}:
            return name
    return None


def solve(premises: Premises, question: Question) -> Outcome:
    """Recompute the answer from the premises (never from a label), with its support certificate and indispensable
    premises. Structural premises (edges, equations, latent declarations) declare the model: deleting an edge
    states a different model rather than withholding information, so they belong to both sets in full. Over the
    probability premises sufficiency (same answer and value) is monotone: the certificate is an inclusion-minimal
    sufficient set by greedy deletion in premise-id order, and a premise is indispensable when the rest of the input
    is not sufficient without it, which is the intersection of all minimal sufficient sets."""
    from dataclasses import replace

    out = _core(premises, question)
    if out.status != "ANSWER":
        return out

    def sufficient(candidate: Premises) -> bool:
        trial = _core(candidate, question, with_band=False)
        return (trial.status, trial.answer, trial.value) == ("ANSWER", out.answer, out.value)

    information = sorted(p.premise_id for p in premises.items if p.kind in ("probability", "joint"))
    structural = sorted(premises.ids() - set(information))  # latent declarations too
    kept = premises
    for pid in information:
        if sufficient(kept.without(pid)):
            kept = kept.without(pid)
    certificate = tuple(sorted(set(structural) | (kept.ids() & set(information))))
    indispensable = tuple(sorted(set(structural) | {pid for pid in information if not sufficient(premises.without(pid))}))
    return replace(out, certificate=certificate, indispensable=indispensable)


def _incompatible(premises: Premises) -> tuple[str, ...] | None:
    """Premise ids that no probability model satisfies together: a joint P(A=a, B=b) above a stated marginal of one of
    its events, or different from a stated conditional times its marginal. None when they are compatible."""
    lookup = _Lookup(premises)
    joints = {p.premise_id: p.value for p in premises.items if p.kind == "joint"}
    for pid, value in sorted(joints.items()):  # a partition: P(A=0, B=b) + P(A=1, B=b) = P(B=b)
        event = dict((var, int(val)) for var, val in (a.split("=") for a in pid[len("j:"):].split(",")))
        if len(event) != 2:
            continue
        for shared, other in (tuple(event), tuple(reversed(tuple(event)))):
            twin = joint_id({shared: event[shared], other: 1 - event[other]})
            if twin not in joints:
                continue
            total, marginal = value + joints[twin], lookup._stated(shared, event[shared], {})
            if total > 1 or (marginal is not None and total != marginal):
                return tuple(sorted({pid, twin} | ({prob_id(shared, 1)} if marginal is not None else set())))
    for p in premises.items:
        if p.kind != "joint":
            continue
        event = dict((var, int(val)) for var, val in (a.split("=") for a in p.premise_id[len("j:"):].split(",")))
        for var, val in event.items():
            marginal = lookup._stated(var, val, {})
            if marginal is None:
                continue
            if p.value > marginal:
                return tuple(sorted({p.premise_id, prob_id(var, 1)}))
            for other, oval in event.items():
                if other == var or len(event) != 2:
                    continue
                conditional = lookup._stated(other, oval, {var: val})
                if conditional is not None and conditional * marginal != p.value:
                    return tuple(sorted({p.premise_id, prob_id(var, 1), prob_id(other, 1, {var: val})}))
    return None


def _core(premises: Premises, question: Question, with_band: bool = True) -> Outcome:
    """Validity first, then the recipe of the query type."""
    import networkx as nx

    if (clash := premises.conflict) is not None:
        return Outcome("ILL_FORMED", reason="CONFLICTING_VALUE", missing=(clash,))
    if (incompatible := _incompatible(premises)) is not None:
        return Outcome("ILL_FORMED", reason="INCOMPATIBLE_PROBABILITIES", missing=incompatible)
    digraph = _digraph(premises)
    reversed_pairs = sorted(f"edge:{a}->{b}" for a, b in digraph.edges if digraph.has_edge(b, a))
    if reversed_pairs:  # an edge and its reversal define no consistent model
        return Outcome("ILL_FORMED", reason="CONFLICTING_DIRECTION", missing=tuple(reversed_pairs))
    if not nx.is_directed_acyclic_graph(digraph):
        cycle = nx.find_cycle(digraph)
        return Outcome("ILL_FORMED", reason="CYCLE", missing=tuple(sorted(f"edge:{a}->{b}" for a, b in cycle)))
    graph = graph_of(premises)
    if question.query_type in ("collider_bias", "backadj", "det-counterfactual"):
        nodes = set(_digraph(premises).nodes)
        needed = {question.treatment, question.outcome} | {v for m in question.methods for v in m}
        if unknown := sorted(needed - nodes):
            return Outcome("NOT_COMPUTABLE", reason="UNKNOWN_VARIABLE", missing=tuple(f"node:{v}" for v in unknown))
    if question.query_type == "collider_bias":
        return Outcome("ANSWER", answer=_collider_bias(premises, question), recipe="graph_path")
    if question.query_type == "backadj":
        return Outcome("ANSWER", answer=_backadj(premises, question), recipe="backdoor_criterion")
    if question.query_type == "det-counterfactual":
        try:
            y = _counterfactual(premises, question)
        except _Missing as missing:
            return Outcome("NOT_COMPUTABLE", recipe="boolean_scm", reason="MISSING_PREMISE", missing=(missing.args[0],))
        except _IllFormed as bad:
            return Outcome("ILL_FORMED", recipe="boolean_scm", reason=bad.args[0], missing=(bad.args[1],))
        return Outcome("ANSWER", answer="yes" if y == question.outcome_value else "no", value=Fraction(y),
                       recipe="boolean_scm")
    key = (question.query_type, graph)
    if key not in RECIPES:  # the stated edges are no CLadder graph of this query type: no recipe to recompute with
        return Outcome("NOT_COMPUTABLE", reason="UNSUPPORTED_GRAPH")
    name, recipe = RECIPES[key]
    try:
        value, absent = _evaluate(premises, name, recipe)
    except _Missing as missing:
        ids = missing.args[0] if isinstance(missing.args[0], tuple) else (missing.args[0],)
        return Outcome("NOT_COMPUTABLE", recipe=name, reason="MISSING_PREMISE", missing=ids)
    except ZeroDivisionError:
        return Outcome("NOT_COMPUTABLE", recipe=name, reason="ZERO_DENOMINATOR")
    except _IllFormed as bad:
        return Outcome("ILL_FORMED", recipe=name, reason=bad.args[0], missing=tuple(sorted(bad.args[1:])))
    answer = decide(question.query_type, question.polarity, value)
    if not with_band:
        return Outcome("ANSWER", answer=answer, value=value, recipe=name)
    band = _band(premises, name, recipe, absent)
    t = threshold(question.query_type)
    conflict = band is None or band[0] <= t <= band[1]
    return Outcome("ANSWER", answer=decide(question.query_type, question.polarity, value), value=value, recipe=name,
                   interval=band, precision_conflict=conflict)


def threshold(query_type: str) -> Fraction:
    """Decision threshold: 'more likely than not' for marginal questions, zero for signed differences."""
    return Fraction(1, 2) if query_type == "marginal" else Fraction(0)


def decide(query_type: str, polarity: str, value: Fraction) -> str:
    """Answer of a numeric question. A value on the threshold answers 'no': the asked strict change is absent.
    An ETT question asks the treated about Y under the other treatment, so its sign is reversed."""
    if polarity in _UP:
        sign = 1
    elif polarity in _DOWN:
        sign = -1
    else:
        raise ValueError(f"no decision function for polarity {polarity!r}")
    if query_type == "ett":
        sign = -sign
    t = threshold(query_type)
    return "yes" if (value > t if sign > 0 else value < t) else "no"
