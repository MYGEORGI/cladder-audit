"""CLadder v1.5 record -> premises and question for the reference solver.

Graph, probabilities and equations come from the structured fields (`graph_id`, `reasoning`, `formal_form`). Only what
those fields lack is read from fixed CLadder templates of the question. The label and the concluding lines of
`reasoning` are never read.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Mapping

from .premises import Premise, Premises, Question, edge_id, joint_id, prob_id

NAMES = Path(__file__).with_name("cladder_names.json")  # story|graph -> {variable name: V symbol}, built from the data

GRAPHS = {
    "IV": (("V1", "X"), ("V2", "X"), ("V1", "Y"), ("X", "Y")),
    "arrowhead": (("X", "V3"), ("V2", "V3"), ("X", "Y"), ("V2", "Y"), ("V3", "Y")),
    "chain": (("X", "V2"), ("V2", "Y")),
    "collision": (("X", "V3"), ("Y", "V3")),
    "confounding": (("V1", "X"), ("V1", "Y"), ("X", "Y")),
    "diamond": (("X", "V3"), ("X", "V2"), ("V2", "Y"), ("V3", "Y")),
    "diamondcut": (("V1", "V3"), ("V1", "X"), ("X", "Y"), ("V3", "Y")),
    "fork": (("X", "Y"), ("V2", "Y")),
    "frontdoor": (("V1", "X"), ("X", "V3"), ("V1", "Y"), ("V3", "Y")),
    "mediation": (("X", "V2"), ("X", "Y"), ("V2", "Y")),
}
_PROB = re.compile(r"^P\(([^)]*)\)\s*=\s*([01]\.\d+)$")
_TYPO = re.compile(r"=(\d)=1\b")


class SolverInputError(ValueError):
    """A record the solver cannot read; the message starts with the reason code."""


def _atoms(text: str) -> dict[str, int]:
    out = {}
    for atom in filter(None, text.split(",")):
        var, _, val = atom.partition("=")
        if val not in ("0", "1"):
            raise SolverInputError(f"UNRECOGNISED_TEMPLATE: probability atom {atom!r}")
        out[var] = int(val)
    return out


def _probability(body: str, value: str) -> Premise:
    head, _, given = body.partition("|")
    event = _atoms(head)
    if len(event) > 1:  # P(Y=1, X=0): a joint probability
        return Premise(joint_id(event), "joint", Fraction(value))
    (var, val), = event.items()
    return Premise(prob_id(var, val, _atoms(given)), "probability", Fraction(value))


def premises_from_fields(graph_id: str, reasoning: str) -> tuple[Premises, tuple[str, ...]]:
    """Edges of the graph table and the probabilities of `reasoning`; notes list repairs made by rule."""
    if graph_id not in GRAPHS:
        raise SolverInputError(f"UNRECOGNISED_TEMPLATE: graph {graph_id!r}")
    edges = GRAPHS[graph_id]
    lines = [line.strip() for line in (reasoning or "").split("\n")]
    if len(lines) > 1 and "->" in lines[1]:
        stated = {tuple(e.split("->")) for e in lines[1].split(",")}
        if stated != set(edges):
            raise SolverInputError(f"GRAPH_MISMATCH: {lines[1]!r} is not the {graph_id} graph")
    items, notes = [Premise(edge_id(a, b), "edge") for a, b in edges], []
    for line in lines:
        m = _PROB.match(line)
        if not m:
            continue
        body = m.group(1).replace(" ", "")
        if _TYPO.search(body):  # generator typo in correlation tasks: P(X=1=1) means P(X=1)
            body = _TYPO.sub(r"=\1", body)
            if "typo P(X=1=1)" not in notes:
                notes.append("typo P(X=1=1)")
        items.append(_probability(body, m.group(2)))
    return Premises.of(items), tuple(notes)


_LET = re.compile(r"^Let (.*)\.$")
_EQUATION = re.compile(r"^(X|Y|V\d) = ((?:not |and |or |X|Y|V\d| )+)$")
_COUNTERFACTUAL = re.compile(r"^Y_\{X=([01])\} = ([01]) \|\s*(.*)$")
_METHODS = re.compile(r"Method 1: (.*?)\s+Method 2: (.*?)(?:\s+To |\s*$)", re.S)
# anchored template phrases: variable names such as "increased price" must not read as a polarity
_POLARITY = ((" positively affect ", "positive"), (" negatively affect ", "negative"),
             (" increase the chance of ", "increase"), (" decrease the chance of ", "decrease"),
             (" increase when ", "increase"), (" decrease when ", "decrease"),
             (" more likely than ", "more"), (" less likely than ", "less"),
             (" more likely to see ", "more"), (" less likely to see ", "less"),
             (" larger when observing ", "larger"), (" smaller when observing ", "smaller"))
_names_table: dict | None = None


def _names_for(story: str, graph: str) -> dict[str, str]:
    global _names_table
    if _names_table is None:
        _names_table = json.loads(NAMES.read_text(encoding="utf-8"))
    return _names_table.get(f"{story}|{graph}", {})


def let_names(reasoning: str) -> dict[str, str]:
    """{variable name (lower case): symbol} from the first line of `reasoning`, 'Let X = husband; V2 = wife; ...'."""
    m = _LET.match((reasoning or "").split("\n")[0].strip())
    if not m:
        return {}
    pairs = (part.split(" = ", 1) for part in m.group(1).split("; ") if " = " in part)
    return {name.strip().lower(): symbol.strip() for symbol, name in pairs}


def repair_equation(var: str, expression: str, parents: Mapping[str, set[str]]) -> str:
    """Generator typo in det-counterfactual: in 'diamond' it writes 'V3 = V2' and in 'diamondcut'
    'V3 = X', naming a variable that is not V3's parent. With one operand and one parent the operand is the parent,
    as the question text states ("CEO ... causes ... director"). Anything else is left for the solver to refuse."""
    words = expression.split()
    operands = [w for w in words if w not in ("not", "and", "or")]
    own = parents.get(var, set())
    if len(set(operands)) == 1 and len(own) == 1 and operands[0] not in own:
        (parent,) = own
        return " ".join(parent if w == operands[0] else w for w in words)
    return expression


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s.strip()]


def _symbol(name: str, names: Mapping[str, str]) -> str:
    key = name.strip().lower()
    if key not in names:
        raise SolverInputError(f"UNRECOGNISED_TEMPLATE: no variable named {name!r}")
    return names[key]


def _polarity(query_type: str, last: str) -> str | None:
    low = f" {last.lower()} "
    if query_type == "det-counterfactual":
        return None
    if query_type == "collider_bias":
        return "not_affects" if " does not affect " in low else "affects" if " affects " in low else None
    if query_type == "backadj":
        return "method1" if "method 1 than method 2" in low else "method2" if "method 2 than method 1" in low else None
    hits = [token for phrase, token in _POLARITY if phrase in low]
    return hits[0] if len(hits) == 1 else None


_QUESTION_CF = re.compile(r"^Would (.+) if (.+) instead of (.+)\?$")
_CONTEXT = ("Imagine a self-contained, hypothetical world with only the following conditions, and without any "
            "unmentioned factors or causal relationships: ")


def _reconcile(sentences: list[str], fields: dict, forms) -> None:
    """the counterfactual question read by the generator's forms must say what
    formal_form says: the intervention ('if A instead of B'), the outcome value ('Would C') and every observed fact
    ('We observed D and E.'). A text the forms do not read is refused, never guessed."""
    if forms is None:
        raise SolverInputError("UNRECOGNISED_TEMPLATE: a det-counterfactual question needs the generator forms")
    from .parsers.cladder_generator import find

    def read(text: str) -> list[tuple[str, int]]:
        found = find(text, forms)
        gaps = {text[a_end:b_start] for (_, a_end, _), (b_start, _, _) in zip(found, found[1:], strict=False)}
        whole = found and found[0][0] == 0 and found[-1][1] == len(text)  # nothing around the forms
        if not whole or gaps - {" and "} or any(f.value not in ("0", "1") for _, _, f in found):
            raise SolverInputError(f"UNRECOGNISED_TEMPLATE: no variable value read in {text!r}")
        return [(f.symbol, int(f.value)) for _, _, f in found]

    def covered(text: str, joins: set[str]) -> bool:  # forms cover the text whole, joined only by `joins`
        found = find(text, forms)
        gaps = {text[a_end:b_start] for (_, a_end, _), (b_start, _, _) in zip(found, found[1:], strict=False)}
        return bool(found) and found[0][0] == 0 and found[-1][1] == len(text) and gaps <= joins

    def whole(i: int, s: str) -> bool:  # every sentence is read whole by its template
        if i == 0:
            if not s.startswith(_CONTEXT):
                return False
            s = s[len(_CONTEXT):]
            m = re.fullmatch(r"(.+) has a direct effect on (.+)\.", s)  # the context opens on an edge, never a fact
            return bool(m) and covered(m.group(1), set()) and covered(m.group(2), {" and ", ", ", ", and "})
        if s.startswith("We observed "):
            return s.endswith(".")  # an assertion; its body is read whole by read() below
        if m := re.fullmatch(r"(.+) has a direct effect on (.+)\.", s):
            return covered(m.group(1), set()) and covered(m.group(2), {" and ", ", ", ", and "})
        if m := re.fullmatch(r"(.+) is unobserved\.", s):
            return covered(m.group(1), set())
        clauses = re.split(r",? and we know that ", s.removeprefix("We know that ").removesuffix("."))
        return s.endswith(".") and all(
            (m := re.fullmatch(r"(.+) causes (.+)", c)) and covered(m.group(1), {" or ", " and "})
            and covered(m.group(2), {" and "}) for c in clauses)

    for i, s in enumerate(sentences[:-1]):
        if not whole(i, s):
            raise SolverInputError(f"UNRECOGNISED_TEMPLATE: sentence {s!r}")
    m = _QUESTION_CF.match(sentences[-1])
    if not m:
        raise SolverInputError(f"UNRECOGNISED_TEMPLATE: question {sentences[-1]!r}")
    (x, value), = fields["intervention"]
    said = {"outcome": read(m.group(1)), "if": read(m.group(2)), "instead of": read(m.group(3)),
            "observed": sorted(atom for s in sentences if s.startswith("We observed ") for atom in read(s[12:-1]))}
    meant = {"outcome": [("Y", fields["outcome_value"])], "if": [(x, value)], "instead of": [(x, 1 - value)],
             "observed": sorted(fields["evidence"])}
    for part in said:
        if said[part] != meant[part]:
            raise SolverInputError(f"FORMAL_FORM_MISMATCH: the question says {part} {said[part]}, "
                                   f"formal_form {meant[part]}")


def record_to_problem(row: Mapping[str, str], forms=None) -> tuple[Premises, Question]:
    """Premises and question of one CLadder record (a csv row). Never reads `label` or the concluding lines. A
    det-counterfactual record needs `forms`, the generator forms of the task (`parsers.cladder_generator`)."""
    query_type, graph, reasoning = row["query_type"], row["graph_id"], row.get("reasoning") or ""
    if reasoning.strip().startswith("nan"):
        reasoning = ""
    base, _ = premises_from_fields(graph, reasoning)
    names = {**_names_for(row["story_id"], graph), **let_names(reasoning)}
    sentences = _sentences(row["prompt"])
    latent = [_symbol(s[: -len(" is unobserved.")], names) for s in sentences if s.endswith(" is unobserved.")]
    items = list(base.items)
    parents: dict[str, set[str]] = {}
    for a, b in GRAPHS[graph]:
        parents.setdefault(b, set()).add(a)
    for line in reasoning.split("\n")[1:]:
        m = _EQUATION.match(line.strip())
        if m:
            items.append(Premise(f"eq:{m.group(1)}", "equation", repair_equation(m.group(1), m.group(2).strip(), parents)))
    polarity = _polarity(query_type, sentences[-1])
    if polarity is None and query_type != "det-counterfactual":
        raise SolverInputError(f"UNRECOGNISED_TEMPLATE: question {sentences[-1]!r}")
    fields: dict = {}
    if query_type == "det-counterfactual":
        m = _COUNTERFACTUAL.match(row["formal_form"].strip())
        if not m:
            raise SolverInputError(f"UNRECOGNISED_TEMPLATE: formal_form {row['formal_form']!r}")
        evidence = tuple(sorted((v.strip(), int(x)) for v, x in (a.split("=") for a in m.group(3).split(",") if "=" in a)))
        fields = {"intervention": (("X", int(m.group(1))),), "outcome_value": int(m.group(2)), "evidence": evidence}
        _reconcile(sentences, fields, forms)
    elif query_type == "backadj":
        m = _METHODS.search(row["prompt"])
        if not m:
            raise SolverInputError("UNRECOGNISED_TEMPLATE: no Method 1 / Method 2 blocks")
        methods = []
        for block in m.groups():
            by = re.search(r"case by case according to (.+?)\.?$", block.strip())
            if by:
                methods.append(tuple(sorted(_symbol(name, names) for name in by.group(1).split(" and "))))
            elif block.strip().endswith("in general."):
                methods.append(())
            else:
                raise SolverInputError(f"UNRECOGNISED_TEMPLATE: method {block!r}")
        fields = {"methods": tuple(methods)}
    question = Question(query_type=query_type, polarity=polarity, **fields)
    return Premises.of(items, latent=latent), question
