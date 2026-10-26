"""Transfer operator T_omega.

T_omega(x) = (D_x, q_x, y*_x): the premises of a CLadder or CaLM-CEI item become sentences in support documents of
their own world, with fresh variable names; distractor documents come ready. I1-I4 are checked per item.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from dataclasses import replace as with_fields
from pathlib import Path
from typing import AbstractSet, Any, Mapping, Optional

from .parsers.cladder_generator import TaskForms, find, replace, skeleton

ROLES = ("EDGE", "LATENT_CONF", "PROB", "CONTEXT", "EQUATION")
INVARIANTS = ("I1", "I2", "I3", "I4")
TEMPLATES = "T-v1"
FRESH_NAMES = "fresh-v1"  # pseudo-words of three consonant-vowel syllables, drawn by seed
_CONSONANTS, _VOWELS = "bdfgklmnprstvz", "aeiou"
_FRESH_NAME = re.compile(f"(?:[{_CONSONANTS}][{_VOWELS}]){{3}}")  # the names FreshNames can give


def world_label(item_id: str, seed: int) -> str:
    """The world of an item in one run: w- and 12 hex digits of SHA-256 over (item id, seed)."""
    return "w-" + hashlib.sha256(f"{item_id}|{seed}".encode()).hexdigest()[:12]


class FreshNames:
    """Fresh variable names for one run: drawn in a seeded order without repeats, so no two worlds share a name, and
    never a word of the source texts (`forbidden`, lower case). 14 * 5 = 70 syllables give 343 000 names."""

    def __init__(self, seed: int, forbidden: AbstractSet[str]):
        syllables = [c + v for c in _CONSONANTS for v in _VOWELS]
        self._size = len(syllables) ** 3
        # the seeded order is the permutation i -> (a*i + b) mod size, a coprime to size: no repeats, no 343 000-long
        # shuffle per item
        rng = random.Random(seed)
        self._a = next(a for a in iter(lambda: rng.randrange(1, self._size), None) if math.gcd(a, self._size) == 1)
        self._b = rng.randrange(self._size)
        self._syllables, self._forbidden, self._next = syllables, {w.lower() for w in forbidden}, 0

    def take(self) -> str:
        n = len(self._syllables)
        while self._next < self._size:
            i = (self._a * self._next + self._b) % self._size
            self._next += 1
            name = self._syllables[i // (n * n)] + self._syllables[i // n % n] + self._syllables[i % n]
            if name not in self._forbidden:
                return name
        raise TransferError("fresh names exhausted")


class TransferError(ValueError):
    """An item that cannot be parsed or transferred; the message names the item."""


_POPULATION = ("wherepartial", "wheresentence")  # "For husbands that set the alarm, ..."


@dataclass(frozen=True)
class Occurrence:
    """One replaced mention: where its fresh form starts in the renamed text, and what it was."""

    start: int
    source: str  # the source text exactly, case included
    symbol: str
    value: str  # "1", "0", "" (the variable name) or "?" (a collision no aligned premise decides)
    kind: str
    fresh: str


def values_of(premise_id: str) -> dict[str, str]:
    """Variable values a premise id states ("p:Y=1|V1=0,X=0" -> Y 1, V1 0, X 0); edges and the rest state none."""
    return dict(re.findall(r"\b(X|Y|V\d)=([01])\b", premise_id))


def _fresh(name: str, value: str, kind: str) -> str:
    """The role scheme: 1 -> N, 0 -> not N, a population -> cases with/without N; a collision keeps one form for both
    values."""
    if value == "?":
        return f"cases with {name} (either)"
    if kind in _POPULATION and value in ("0", "1"):
        return f"cases {'with' if value == '1' else 'without'} {name}"
    return f"not {name}" if value == "0" else name


def rename(text: str, forms: TaskForms, bijection: Mapping[str, str],
           values: Mapping[str, str] | None = None) -> tuple[str, tuple[Occurrence, ...]]:
    """Every mention of a variable form, longest first at word bounds, becomes its fresh form; the journal keeps each
    source form so `restore` gives the text back byte for byte. `values`: those of the premise the text states, for
    the collisions. The generic `subject` noun is never a form, so it stays."""
    out, journal, pos = [], [], 0
    for start, end, form in find(text, forms):
        if form.symbol not in bijection:
            raise TransferError(f"no fresh name for {form.symbol} in {text!r}")
        out.append(text[pos:start])
        fresh = _fresh(bijection[form.symbol], form.value, form.kind)
        value = (values or {}).get(form.symbol, "?") if form.value == "?" else form.value
        journal.append(Occurrence(sum(map(len, out)), text[start:end], form.symbol, value, form.kind, fresh))
        out.append(fresh)
        pos = end
    out.append(text[pos:])
    return "".join(out), tuple(journal)


def restore(text: str, journal: tuple[Occurrence, ...]) -> str:
    """The source text from a renamed one and its journal; a fresh form not where the journal puts it, or a malformed
    occurrence, is refused."""
    if not all(isinstance(o, Occurrence) and type(o.start) is int and isinstance(o.source, str)
               and isinstance(o.fresh, str) for o in journal):
        raise TransferError(f"malformed journal for {text!r}")
    for occ in sorted(journal, key=lambda o: o.start, reverse=True):
        if text[occ.start:occ.start + len(occ.fresh)] != occ.fresh:
            raise TransferError(f"journal does not fit {text!r}: no {occ.fresh!r} at {occ.start}")
        text = text[:occ.start] + occ.source + text[occ.start + len(occ.fresh):]
    return text


@dataclass(frozen=True)
class ItemPremise:
    premise_id: str
    role: str
    text: str  # the atomic sentence in the source names

    def __post_init__(self):
        if self.role not in ROLES:
            raise TransferError(f"{self.premise_id}: role {self.role!r} is not one of {ROLES}")


@dataclass(frozen=True)
class Item:
    item_id: str
    premises: tuple[ItemPremise, ...]
    question: str  # q_x: the prompt without its premise sentences
    reference: str  # y*_x of the dataset: yes or no
    scenario: str
    rung: int
    solver: Any = None  # (Premises, Question) for CLadder, None for CaLM-CEI
    forms: Optional[TaskForms] = None  # the variable forms renaming replaces (generator forms or CEI vertices)

    def __post_init__(self):
        ids = [p.premise_id for p in self.premises]
        if not ids:
            raise TransferError(f"{self.item_id}: no premises")
        if len(set(ids)) != len(ids):
            raise TransferError(f"{self.item_id}: duplicate premise ids")
        if self.reference not in ("yes", "no"):
            raise TransferError(f"{self.item_id}: reference {self.reference!r} is not yes or no")


@dataclass(frozen=True)
class Omega:
    seed: int
    templates: str = TEMPLATES
    k: Optional[int] = None  # None: one premise per document
    distractors: Optional[str] = None  # id of a distractor set

    def __post_init__(self):
        if self.k is not None and self.k < 1:
            raise TransferError(f"k = {self.k}: must be None or at least 1")


@dataclass(frozen=True)
class Document:
    doc_id: str
    world: str
    kind: str  # support or distractor
    sentences: tuple[tuple[Optional[str], str], ...]  # (premise id or None, text)

    def __post_init__(self):
        if self.kind not in ("support", "distractor"):
            raise TransferError(f"{self.doc_id}: kind {self.kind!r}")


@dataclass(frozen=True)
class Violation:
    item_id: str
    invariant: str
    detail: str

    def __post_init__(self):
        if self.invariant not in INVARIANTS:
            raise TransferError(f"{self.item_id}: invariant {self.invariant!r}")


TEMPLATES_SHA256 = Path(__file__).with_name("cladder_templates.sha256").read_text(encoding="utf-8").split()[0]
PARSERS = {"cladder": "cladder-v1", "calm-cei": "calm-cei-v1"}


@dataclass(frozen=True)
class TransferredItem:
    item: Item
    omega: Omega
    world: str
    documents: tuple[Document, ...]  # support and distractor documents, order shuffled by seed
    question: str  # q_x renamed
    bijection: tuple[tuple[str, str], ...]  # symbol -> fresh name
    journals: tuple[tuple[str, tuple[Occurrence, ...]], ...]  # premise id or "question" -> occurrences
    description: dict  # the manifest's `transfer` entry for this item


def transfer(item: Item, omega: Omega, distractors: tuple[Document, ...] = (), *,
             names: FreshNames) -> TransferredItem:
    """T_omega(x): rename every sentence and the question, deal the premises shuffled by
    seed round-robin into min(k, |Pi|) support documents of the item's world, add the distractors, shuffle the
    document order. `names` is required: one FreshNames for the whole run, so no two worlds share a name; a new FreshNames(seed, forbidden) per call gives the
    same result for the same (item, omega)."""
    if item.forms is None:
        raise TransferError(f"{item.item_id}: no variable forms to rename")
    world = world_label(item.item_id, omega.seed)
    bijection = {s: names.take() for s in sorted({f.symbol for f in item.forms.forms})}
    renamed, journals = {}, []
    for p in item.premises:
        renamed[p.premise_id], journal = rename(p.text, item.forms, bijection, values_of(p.premise_id))
        journals.append((p.premise_id, journal))
    question, journal = rename(item.question, item.forms, bijection)
    journals.append(("question", journal))
    rng = random.Random(f"{omega.seed}|{item.item_id}")
    ids = [p.premise_id for p in item.premises]
    rng.shuffle(ids)
    k = min(omega.k or len(ids), len(ids))
    documents = [Document("doc-" + hashlib.sha256(f"{world}|{i}".encode()).hexdigest()[:12], world, "support",
                          tuple((pid, renamed[pid]) for pid in ids[i::k])) for i in range(k)]
    documents += list(distractors)
    rng.shuffle(documents)
    description = {"item_id": item.item_id, "seed": omega.seed, "templates": omega.templates, "k": omega.k,
                   "distractors": omega.distractors, "templates_sha256": TEMPLATES_SHA256, "fresh_names": FRESH_NAMES,
                   "parser": PARSERS[item.item_id.split(":")[0]]}
    return TransferredItem(item, omega, world, tuple(documents), question, tuple(sorted(bijection.items())),
                           tuple(journals), description)


def _restored(text: str, journal: Optional[tuple[Occurrence, ...]]) -> Optional[str]:
    """The source text, or None when there is no journal or it does not fit (a mutation, never an error of check)."""
    try:
        return None if journal is None else restore(text, journal)
    except TransferError:
        return None


def check(t: TransferredItem, item: Item) -> list[Violation]:
    """I1-I3 of one transferred item against its source `item` (contract properties 4-6); I4 is
    `check_manifest_transfer`. The source item is passed apart from t.item, so a changed reference has something to
    differ from. Sorted by invariant and detail, so a report is byte-identical across runs.

    I1: the premise ids of the support sentences are exactly those of the item, each once, and every sentence restored
    by its journal is its premise's source sentence byte for byte. Renaming never paraphrases and the
    parser tied each source sentence to its premise, so this is stronger than parsing the content again.

    A transferred item whose fields are not of their declared shape (None, a wrong container) cannot be read back,
    so it is Violation("I3", "malformed"), never an error."""
    try:
        return _check(t, item)
    except MALFORMED:
        return [Violation(item.item_id, "I3", "malformed")]


MALFORMED = (TypeError, AttributeError, KeyError, ValueError, IndexError)  # what reading a damaged artifact raises


def _check(t: TransferredItem, item: Item) -> list[Violation]:
    out: list[Violation] = []
    texts = {p.premise_id: p.text for p in item.premises}
    journals = dict(t.journals)
    support = [(pid, text) for d in t.documents if d.kind == "support" for pid, text in d.sentences]
    out += [Violation(item.item_id, "I1", pid) for pid in set(texts) - {pid for pid, _ in support}]
    seen = set()
    for pid, text in support:
        if pid in seen or pid not in texts or _restored(text, journals.get(pid)) != texts[pid]:
            out.append(Violation(item.item_id, "I1", str(pid)))
        seen.add(pid)
    out += _names_in_distractors(t, item)
    out += [Violation(item.item_id, "I3", d) for d in _renaming_faults(t, item, [text for _, text in support])]
    return sorted(out, key=lambda v: (v.invariant, v.detail))


TEMPLATE_LIST = frozenset(json.loads(Path(__file__).with_name("cladder_templates.json").read_text(encoding="utf-8"))
                          ["templates"])


def _renaming_faults(t: TransferredItem, item: Item, support: list[str]) -> list[str]:
    """I3: the question restores byte for byte and the reference is the
    item's; the bijection is one of fresh names; every journal is the renaming of its whole source text, so each
    occurrence keeps its symbol, its value and its fresh form; no source
    form is left in a support sentence or the question; and every CLadder source text but the context reduces to a
    frozen template. The last check does not use the dictionary, so it sees a form the dictionary lost."""
    out = []
    if _restored(t.question, dict(t.journals).get("question")) != item.question:
        out.append("question")
    if t.item.reference != item.reference:
        out.append("reference")
    # untransfer reads t.item.solver and t.item.forms, so the embedded item is the source item
    if with_fields(t.item, reference=item.reference) != item:
        out.append("item")
    bijection = dict(t.bijection)
    names = list(bijection.values())
    # a mapping that merges two symbols reads back consistently, so it is checked on its own:
    # distinct names of the fresh-v1 vocabulary, so no fresh form of one symbol ("not N") is a form of another
    if (len(t.bijection) != len(bijection) or set(bijection) != {f.symbol for f in item.forms.forms}
            or len(set(names)) != len(names) or not all(_FRESH_NAME.fullmatch(n) for n in names)):
        out.append("bijection")
    # every journal is the one renaming the whole source text with the item's forms and this bijection gives: an entry
    # checked alone let a shorter form inside a longer mention pass
    sources = {p.premise_id: (p.text, values_of(p.premise_id)) for p in item.premises}
    sources["question"] = (item.question, None)
    for key, journal in t.journals:
        try:
            expected = rename(sources[key][0], item.forms, bijection, sources[key][1])[1] if key in sources else None
        except TransferError:
            expected = None
        if journal != expected:
            out.append(f"journal {key}")
    out += [form.text for text in support + [t.question] for _, _, form in find(text, item.forms)]
    if item.item_id.startswith("cladder:"):
        texts = [p.text for p in item.premises if p.role != "CONTEXT"]
        residuals = [skeleton(replace(x, item.forms))
                     for x in texts + [s for s in re.split(r"(?<=[.?!])\s+", item.question) if s]]
        out += [r for r in residuals if r not in TEMPLATE_LIST]
    return out


_TOKEN = re.compile(r"[\w-]+")


def _names_in_distractors(t: TransferredItem, item: Item) -> list[Violation]:
    """I2: no distractor sentence names a word of V_x, the source forms, the fresh names and the CEI
    vertices. Matched as token sequences, so a name inside another word does not count; source forms and fresh names
    ignore case, a CEI vertex is an exact upper-case token ("a" is not "A"). The generic `subject` is never a form."""
    names = [(f.text, f.kind == "vertex") for f in item.forms.forms] + [(n, False) for _, n in t.bijection]
    out = []
    for d in t.documents:
        if d.kind != "distractor":
            continue
        for _, text in d.sentences:
            tokens = _TOKEN.findall(text)
            lower = [w.lower() for w in tokens]
            for name, exact in names:
                words = _TOKEN.findall(name if exact else name.lower())
                seq = tokens if exact else lower
                if any(seq[i:i + len(words)] == words for i in range(len(seq) - len(words) + 1)):
                    out.append(Violation(item.item_id, "I2", name))
    return out


def check_manifest_transfer(manifest: Mapping[str, Any], transferred) -> list[Violation]:
    """I4 against the manifest as written: `extensions.transfer.items` holds each
    transferred item's description, equal field by field, and the versions in it are those of this code. A missing
    entry is one Violation under the run id; a stale manifest (same ids, other seed or k) differs in those fields.
    The expected entry comes from the Omega used, the item and this code, never from the description alone; a manifest of another shape is a violation, never an error."""
    extensions = manifest.get("extensions")
    entry = extensions.get("transfer") if isinstance(extensions, Mapping) else None
    if not isinstance(entry, Mapping):
        return [Violation(str(manifest.get("run_id")), "I4", "transfer")]
    items = entry.get("items")
    items = items if isinstance(items, Mapping) else {}
    out = []
    for t in transferred:
        tid = t.item.item_id
        want = {"item_id": tid, "seed": t.omega.seed, "templates": TEMPLATES, "k": t.omega.k,
                "distractors": t.omega.distractors, "templates_sha256": TEMPLATES_SHA256, "fresh_names": FRESH_NAMES,
                "parser": PARSERS[tid.split(":")[0]]}
        got = items.get(tid)
        if not isinstance(got, Mapping):
            out.append(Violation(tid, "I4", "missing"))
            continue
        if not isinstance(t.description, Mapping):
            out.append(Violation(tid, "I4", "description"))
            continue
        keys = set(want) | set(got) | set(t.description)
        # an absent field is not a null one, and 1, 1.0 and True are not one seed: membership
        # and type are checked before the value
        out += [Violation(tid, "I4", k) for k in sorted(keys)
                if k not in want or k not in got or k not in t.description
                or not _same(got[k], want[k]) or not _same(t.description[k], want[k])]
    return out


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def untransfer(t: TransferredItem) -> tuple[Any, Any]:
    """CLadder only: the solver's (Premises, Question) from the transferred text. Each support sentence
    is restored by its journal and parsed again; the question restores byte for byte, so the record's question
    stands for it (I3)."""
    from .parsers.cladder import solver_premises  # the parser imports this module

    if not t.item.item_id.startswith("cladder:") or t.item.solver is None:
        raise TransferError(f"{t.item.item_id}: untransfer reads CLadder items only")
    journals = dict(t.journals)
    if restore(t.question, journals["question"]) != t.item.question:
        raise TransferError(f"{t.item.item_id}: the question does not restore")
    sentences = [(restore(text, journals[pid]), journals[pid])
                 for d in t.documents if d.kind == "support" for pid, text in d.sentences]
    return solver_premises(sentences, t.item.forms, t.item.item_id), t.item.solver[1]
