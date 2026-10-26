"""Surface forms of CLadder v1.5 variables from the CLadder generator itself.

Sources at causalNLP/cladder 4e54416 (`scripts/fetch_cladder.py`):
- `data/cladder-v1-meta-models.json`: per model the `variable_mapping` (Xname, X1, X0, V2name, ...);
- `assets/stories/*.yml`: per story letter (W, X, Y, Z, ...) the forms name, noun, sentence, sentence_condition,
  wherepartial, wheresentence (and subject, a generic noun that is never a variable mention);
- `assets/anticommonsense.yml`: the rewrites an anticommonsense variant may carry (`verbal/anticommonsense.py`).
The ETT question uses `_sentence_condition.split(" instead of")[0]` (`queries/ett.py`).

A task links to exactly one model by and to exactly one story variant by
equal phrases with the variable_mapping; anything else is a GeneratorError. A form the generator gives to both values
of one variable is a collision: its slot carries value "?" and the task is flagged.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

_FIELD = re.compile(r"^([A-Z])([01])?_?(.*)$")
_SYMBOL = re.compile(r"^(X|Y|V\d)")


class GeneratorError(ValueError):
    """A task without exactly one generator model or story variant; the message names the task."""


@dataclass(frozen=True)
class Form:
    symbol: str  # X, Y, V1, ...
    value: str  # "1", "0", "?" (a collision of both values) or "" (the variable name)
    kind: str  # name, noun, sentence, sentence_condition, sentence_condition_chop, wherepartial, wheresentence, value
    text: str

    @property
    def slot(self) -> str:
        return f"{self.symbol}{self.value}:{self.kind}"


@dataclass(frozen=True)
class TaskForms:
    forms: tuple[Form, ...]  # longest text first
    collisions: tuple[str, ...]  # texts the generator gives to both values of one variable


def sentences(prompt: str) -> list[str]:
    head, sep, rest = prompt.strip().partition("relationships: ")
    return [head + sep.strip()] + [s for s in re.split(r"(?<=[.?!])\s+", rest) if s]


def background(sents: list[str]) -> str:
    keep = [s for s in sents[1:] if " has a direct effect on " in s or s.endswith(" is unobserved.")]
    return " ".join([sents[0]] + keep).strip()


class Generator:
    def __init__(self, meta: list[dict], stories: Mapping[str, dict], anti: dict):
        self.models: dict[tuple, set[str]] = defaultdict(set)
        for m in meta:
            key = (m["story_id"], m["graph_id"], m["background"].strip())
            self.models[key].add(json.dumps(m["variable_mapping"], sort_keys=True))
        self.stories, self.anti = dict(stories), anti

    @classmethod
    def load(cls, root: Path) -> "Generator":
        """`root` holds cladder-v1-meta-models.json (or data/...), stories/ (or assets/stories/), anticommonsense.yml."""
        def first(*paths: Path) -> Path:
            return next(p for p in paths if p.exists())

        meta = json.loads(first(root / "cladder-v1-meta-models.json", root / "data" / "cladder-v1-meta-models.json")
                          .read_text(encoding="utf-8"))
        story_dir = first(root / "stories", root / "assets" / "stories")
        stories = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(story_dir.glob("*.yml"))}
        anti = yaml.safe_load(first(root / "anticommonsense.yml", root / "assets" / "anticommonsense.yml")
                              .read_text(encoding="utf-8"))
        return cls(meta, stories, anti)

    def _variants(self, story: dict) -> list[dict]:
        out = [story]
        plurals, subjects, variables = (self.anti.get(k, {}) for k in ("plurals", "subjects", "variables"))
        for key in story:
            if not key.endswith("subject"):
                continue
            var, subject = key[: -len("subject")], story[key]
            plural = plurals.get(subject, f"{subject}s")
            for choice in subjects.get(var, {}).get(subject, []):
                fixes = variables.get(var, {}).get(choice, {})
                change = dict(story)
                change.update({k: v.format(subject=subject, plural=plural) for k, v in fixes.items() if k in story})
                out.append(change)
        return out

    @staticmethod
    def _link(story: dict, mapping: Mapping[str, str]) -> dict[str, str] | None:
        phrases = {p.strip().lower(): _SYMBOL.match(k).group(1) for k, p in mapping.items()}
        found: dict[str, set[str]] = defaultdict(set)
        for key, value in story.items():
            m = _FIELD.match(key)
            if m and isinstance(value, str) and value.strip().lower() in phrases:
                found[m.group(1)].add(phrases[value.strip().lower()])
        if any(len(s) != 1 for s in found.values()):
            return None
        return {letter: next(iter(s)) for letter, s in found.items()}

    def task_forms(self, row: Mapping[str, str]) -> TaskForms:
        sents = sentences(row["prompt"])
        found = self.models.get((row["story_id"], row["graph_id"], background(sents)), set())
        if len(found) != 1:
            raise GeneratorError(f"cladder:{row['id']}: {len(found)} generator models")
        mapping = json.loads(next(iter(found)))
        story = self.stories.get(row["story_id"])
        if story is None:
            raise GeneratorError(f"cladder:{row['id']}: no story {row['story_id']!r}")
        symbols = {_SYMBOL.match(k).group(1) for k in mapping}
        tables = set()
        for variant in self._variants(story):
            letters = self._link(variant, mapping)
            if letters is not None and set(letters.values()) == symbols:
                tables.add(tuple(sorted(self._raw_forms(variant, letters, mapping))))
        if len(tables) != 1:
            raise GeneratorError(f"cladder:{row['id']}: {len(tables)} story variants")
        return _resolve(next(iter(tables)))

    @staticmethod
    def _raw_forms(story: dict, letters: Mapping[str, str], mapping: Mapping[str, str]) -> set[tuple[str, str, str, str]]:
        out = set()
        for key, phrase in mapping.items():
            sym = _SYMBOL.match(key).group(1)
            rest = key[len(sym):]
            out.add((sym, "" if rest == "name" else rest, "name" if rest == "name" else "value", phrase.strip()))
        for key, value in story.items():
            m = _FIELD.match(key)
            if not (m and isinstance(value, str) and m.group(1) in letters) or m.group(3) == "subject":
                continue
            sym, val, kind = letters[m.group(1)], m.group(2) or "", m.group(3) or "name"
            out.add((sym, val, kind, value.strip()))
            if kind == "sentence_condition":
                out.add((sym, val, "sentence_condition_chop", value.split(" instead of")[0].strip()))
        return {f for f in out if f[3]}


def _resolve(raw: tuple[tuple[str, str, str, str], ...]) -> TaskForms:
    """One Form per text: both values of one symbol -> value '?' (a collision); a name equal to a value form keeps the
    value (renaming maps value 1 and the name to the same fresh name); otherwise the most specific kind."""
    by_text: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for sym, val, kind, text in raw:
        by_text[text.lower()].add((sym, val, kind))
    texts = {text.lower(): text for *_, text in raw}
    forms, collisions = [], []
    for low, slots in by_text.items():
        symbols = {s for s, _, _ in slots}
        if len(symbols) != 1:
            raise GeneratorError(f"form {texts[low]!r} names {sorted(symbols)}")
        (sym,) = symbols
        values = {v for _, v, _ in slots if v}
        value = "?" if len(values) > 1 else next(iter(values), "")
        if value == "?":
            collisions.append(texts[low])
        kinds = sorted(k for _, v, k in slots if v == value or value == "?" or not values)
        forms.append(Form(sym, value, kinds[0], texts[low]))
    forms.sort(key=lambda f: (-len(f.text), f.text))
    return TaskForms(tuple(forms), tuple(sorted(collisions)))


def find(text: str, forms: TaskForms) -> list[tuple[int, int, Form]]:
    """Non-overlapping mentions of the task's forms in `text`, longest form first, at word bounds, in text order.
    A CaLM-CEI vertex matches its case exactly; generator forms ignore case."""
    taken: list[tuple[int, int, Form]] = []
    for form in forms.forms:
        flags = "" if form.kind == "vertex" else "(?i)"
        for m in re.finditer(flags + r"(?<![\w-])" + re.escape(form.text) + r"(?![\w-])", text):
            if all(m.end() <= a or m.start() >= b for a, b, _ in taken):
                taken.append((m.start(), m.end(), form))
    return sorted(taken, key=lambda t: t[0])


def replace(text: str, forms: TaskForms) -> str:
    """Every form as its slot `[symbol value:kind]`, longest first, at word bounds; numbers as #."""
    for form in forms.forms:
        text = re.sub(r"(?i)(?<![\w-])" + re.escape(form.text) + r"(?![\w-])", f"[{form.slot}]", text)
    return re.sub(r"(?<![\w.])-?\d+(?:\.\d+)?%?", "#", text)  # numbers only, never the digit of a slot like V2


def skeleton(residual: str) -> str:
    """A residual with symbol and value erased: the grammatical template (`[S:kind]` slots)."""
    return re.sub(r"\[(?:X|Y|V\d)[01?]?:([a-z_]+)\]", r"[S:\1]", residual)
