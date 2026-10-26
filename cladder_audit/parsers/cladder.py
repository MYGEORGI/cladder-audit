"""CLadder v1.5 task -> Item of the transfer operator.

The premises are those of the solver (`record_to_problem`), each paired with the prompt sentence that
states it; a sentence with several edges or equation effects gives one atomic sentence each. Prompt sentence shapes: CONTEXT (the opening), EDGE, LATENT ("... is unobserved."), PROB (probability or correlation), EQUATION
("... causes ..."), EVIDENCE ("We observed ...") and METHOD ("Method 1: ..."); EVIDENCE, METHOD and the final question
form q_x. Variables are recognised by the generator's forms (`cladder_generator`), never by guessing.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Mapping

from ..cladder import _PROB, _TYPO, SolverInputError, _probability, record_to_problem
from ..premises import Premise, Premises, joint_id, prob_id
from ..transfer import Item, ItemPremise, TransferError
from .cladder_generator import Generator, GeneratorError, TaskForms, find, sentences

_EDGE = " has a direct effect on "


def _symbols(text: str, forms: TaskForms) -> list[str]:
    return [form.symbol for _, _, form in find(text, forms)]


def _prob_ids(reasoning: str) -> list[tuple[str, int]]:
    """(premise id, value in percent) in the order of the `reasoning` lines."""
    out = []
    for line in reasoning.split("\n"):
        m = _PROB.match(line.strip())
        if m:
            premise = _probability(_TYPO.sub(r"=\1", m.group(1).replace(" ", "")), m.group(2))
            out.append((premise.premise_id, round(premise.value * 100)))
    return out


def _take_probability(sentence: str, remaining: list[tuple[str, int]], forms: TaskForms, tid: str) -> str:
    """The id whose value equals the sentence's percentage and whose variables are those the sentence names; among
    equal candidates the first in reasoning order."""
    m = re.search(r" is (\d+)%\.$", sentence)
    shape = re.match(r"^For (.+), the probability of (.+) is \d+%\.$", sentence)
    condition, event = (shape.group(1), shape.group(2)) if shape else ("", sentence)
    given_mentions = [(f.symbol, f.value) for _, _, f in find(condition, forms)]
    event_mentions = [(f.symbol, f.value) for _, _, f in find(event, forms)]
    for i, (pid, value) in enumerate(remaining):
        head, _, given = pid[2:].partition("|")  # event atoms and conditioning atoms apart
        if (m and value == int(m.group(1)) and _same_atoms(event_mentions, re.findall(r"([A-Z]\d*)=([01])", head))
                and _same_atoms(given_mentions, re.findall(r"([A-Z]\d*)=([01])", given))):
            del remaining[i]
            return pid
    raise TransferError(f"{tid}: probability sentence {sentence!r} matches no premise by value and variables")


def _same_atoms(mentions: list[tuple[str, str]], atoms: list[tuple[str, str]]) -> bool:
    """The sentence names exactly the premise's variables with the premise's values. A generator
    collision (value '?') admits either value; a bare name admits none."""
    if sorted(s for s, _ in mentions) != sorted(s for s, _ in atoms):
        return False
    wanted = dict(atoms)
    return all(v == "?" or v == wanted[s] for s, v in mentions)


def question_parts(item: Item) -> list[str]:
    """The sentences of q_x (evidence, methods, the question)."""
    return [s for s in re.split(r"(?<=[.?!])\s+", item.question) if s]


def _edges(sentence: str, forms: TaskForms, tid: str) -> list[tuple[str, str]]:
    cause, _, effects = sentence[:-1].partition(_EDGE)
    (a,) = _symbols(cause, forms) or [None]
    parts = [p for p in re.split(r",? and |, ", effects) if p]
    out = []
    for part in parts:
        symbols = _symbols(part, forms)
        if a is None or len(symbols) != 1:
            raise TransferError(f"{tid}: edge sentence {sentence!r} does not resolve to variables")
        out.append((f"edge:{a}->{symbols[0]}", f"{cause}{_EDGE}{part}."))
    return out


def _literal(text: str, forms: TaskForms, tid: str) -> str:
    found = find(text, forms)
    if len(found) != 1 or found[0][2].value not in ("0", "1"):
        raise TransferError(f"{tid}: equation operand {text!r} is not one variable value")
    form = found[0][2]
    return form.symbol if form.value == "1" else f"not {form.symbol}"


def _truth(expression: str, world: dict[str, bool]) -> bool:
    """A solver equation ('X', 'not V1', 'X or V3', 'V1 and not V2') under an assignment."""
    tokens = expression.replace("(", " ( ").replace(")", " ) ").split()
    code = " ".join(t if t in ("not", "and", "or", "(", ")") else str(world[t]) for t in tokens)
    return bool(eval(code, {"__builtins__": {}}))  # tokens are only booleans and the three operators


def _same_function(a: str, b: str) -> bool:
    names = sorted(set(re.findall(r"\b(?:X|Y|V\d)\b", f"{a} {b}")))
    for bits in range(2 ** len(names)):
        world = {n: bool(bits >> i & 1) for i, n in enumerate(names)}
        if _truth(a, world) != _truth(b, world):
            return False
    return True


def _negated(expression: str) -> str:
    """not (a op b ...) in the solver's syntax, without brackets (De Morgan; one operator per cause)."""
    op = " and " if " or " in expression else " or "
    literals = expression.split(" or " if op == " and " else " and ")
    return op.join(x.removeprefix("not ") if x.startswith("not ") else f"not {x}" for x in literals)


def _equation_clauses(sentence: str, forms: TaskForms, tid: str) -> list[tuple[str, str, str]]:
    """(effect symbol, the equation the text states in the solver's syntax, atomic sentence) per clause."""
    out = []
    for clause in re.split(r",? and we know that ", sentence[:-1]):
        clause = re.sub(r"^We know that ", "", clause)
        cause, sep, effects = clause.partition(" causes ")
        if not sep:
            raise TransferError(f"{tid}: equation clause {clause!r}")
        # operands are generator forms (a form may contain "or": "college degree or higher"); only the gaps between
        # them are operators
        found = find(cause, forms)
        gaps = [cause[a_end:b_start] for (_, a_end, _), (b_start, _, _) in zip(found, found[1:], strict=False)]
        if (not found or found[0][0] != 0 or found[-1][1] != len(cause)
                or any(g not in (" or ", " and ") for g in gaps) or len(set(gaps)) > 1):
            raise TransferError(f"{tid}: equation cause {cause!r} is not variable values joined by one operator")
        literals = [_literal(cause[a:b], forms, tid) for a, b, _ in found]
        expression = gaps[0].strip().join(f" {x} " for x in literals).strip() if gaps else literals[0]
        expression = re.sub(r"\s+", " ", expression)
        for part in [p for p in re.split(r" and ", effects) if p]:
            effect = _literal(part, forms, tid)
            symbol = effect.removeprefix("not ")
            out.append((symbol, expression if effect == symbol else _negated(expression), f"{cause} causes {part}."))
    return out


def _equations(sentence: str, forms: TaskForms, tid: str, stated: Mapping[str, str]) -> list[tuple[str, str]]:
    """Atomic equation sentences; each must state the solver's equation for its effect. A source
    whose text says another function than its reasoning field is refused as source_equation_mismatch, never fixed."""
    out = []
    for symbol, text_says, atomic in _equation_clauses(sentence, forms, tid):
        pid = f"eq:{symbol}"
        if pid not in stated or not _same_function(text_says, stated[pid]):
            raise TransferError(f"{tid}: source_equation_mismatch: the text says {symbol} = {text_says}, "
                                f"the reasoning field {stated.get(pid)!r}")
        out.append((pid, atomic))
    return out


def _read_probability(text: str, journal, forms: TaskForms, tid: str) -> Premise:
    """A probability sentence in source names -> the solver premise it states: atoms and value from the text; a
    collision takes the value its journal occurrence keeps."""
    m = re.match(r"^(?:For .+, the|The(?: overall)?) probability of .+ is (\d+)%\.$", text)
    found = find(text, forms)
    if not m or len(found) != len(journal):
        raise TransferError(f"{tid}: probability sentence {text!r}")
    event_at = text.index("probability of ")
    event, given = {}, {}
    for (start, _, form), occ in zip(found, sorted(journal, key=lambda o: o.start), strict=True):
        value = occ.value if form.value == "?" else form.value
        if value not in ("0", "1"):
            raise TransferError(f"{tid}: no value for {form.symbol} in {text!r}")
        (event if start > event_at else given)[form.symbol] = int(value)
    value = Fraction(int(m.group(1)), 100)
    if value > 1:
        raise TransferError(f"{tid}: probability above 100 % in {text!r}")
    if len(event) > 1:
        return Premise(joint_id(event), "joint", value)
    ((var, val),) = event.items()
    return Premise(prob_id(var, val, given), "probability", value)


def solver_premises(sentences, forms: TaskForms, tid: str) -> Premises:
    """The solver's premises read from support sentences in source names, each with its journal:
    content only, never the premise ids the sentences travel with."""
    items, latent = [], []
    for text, journal in sentences:
        if text.startswith("Imagine ") or " the correlation between " in text:
            continue  # the context and the correlation are not solver premises
        if _EDGE in text:
            items += [Premise(pid, "edge") for pid, _ in _edges(text, forms, tid)]
        elif text.endswith(" is unobserved."):
            latent += _symbols(text[: -len(" is unobserved.")], forms)
        elif " probability of " in text:
            items.append(_read_probability(text, journal, forms, tid))
        elif " causes " in text:
            items += [Premise(f"eq:{s}", "equation", e) for s, e, _ in _equation_clauses(text, forms, tid)]
        else:
            raise TransferError(f"{tid}: unrecognised support sentence {text!r}")
    return Premises.of(items, latent=latent)


def parse(row: Mapping[str, str], generator: Generator) -> Item:
    tid = f"cladder:{row['id']}"
    try:
        forms = generator.task_forms(row)
        solver = record_to_problem(row, forms)
    except (GeneratorError, SolverInputError) as exc:
        raise TransferError(f"{tid}: {exc}") from exc
    sents = sentences(row["prompt"])
    premises = [ItemPremise("ctx:0", "CONTEXT", sents[0])]
    probs = _prob_ids(row.get("reasoning") or "")
    question: list[str] = []
    for s in sents[1:-1]:
        if s.startswith(("We observed ", "Method 1: ", "Method 2: ")):
            question.append(s)
        elif _EDGE in s:
            premises += [ItemPremise(pid, "EDGE", text) for pid, text in _edges(s, forms, tid)]
        elif s.endswith(" is unobserved."):
            (symbol,) = _symbols(s[: -len(" is unobserved.")], forms) or [None]
            if symbol is None:
                raise TransferError(f"{tid}: latent sentence {s!r}")
            premises.append(ItemPremise(f"latent:{symbol}", "LATENT_CONF", s))
        elif " the correlation between " in s:
            premises.append(ItemPremise("corr:0", "PROB", s))
        elif " probability of " in s:
            premises.append(ItemPremise(_take_probability(s, probs, forms, tid), "PROB", s))
        elif " causes " in s:
            stated = {p.premise_id: p.value for p in solver[0].items if p.kind == "equation"}
            premises += [ItemPremise(pid, "EQUATION", text) for pid, text in _equations(s, forms, tid, stated)]
        else:
            raise TransferError(f"{tid}: unrecognised sentence {s!r}")
    question.append(sents[-1])
    parsed = {p.premise_id for p in premises}
    needed = {p.premise_id for p in solver[0].items}
    if not needed <= parsed:
        raise TransferError(f"{tid}: premises not stated in the prompt: {sorted(needed - parsed)}")
    return Item(item_id=tid, premises=tuple(premises), question=" ".join(question), reference=row["label"],
                scenario=row["query_type"], rung=int(row["rung"]), solver=solver, forms=forms)
