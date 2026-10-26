"""Premises, questions and outcomes of the reference solver.

Probabilities are exact fractions; premise ids are built from content, so the order of premises never matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Mapping, Optional

KINDS = ("edge", "probability", "joint", "equation", "latent", "assumption")


def edge_id(parent: str, child: str) -> str:
    return f"edge:{parent}->{child}"


def _assign(values: Mapping[str, int]) -> str:
    return ",".join(f"{var}={values[var]}" for var in sorted(values))


def prob_id(var: str, value: int, given: Optional[Mapping[str, int]] = None) -> str:
    return f"p:{var}={value}" + (f"|{_assign(given)}" if given else "")


def joint_id(values: Mapping[str, int]) -> str:
    return f"j:{_assign(values)}"


@dataclass(frozen=True)
class Premise:
    premise_id: str
    kind: str
    value: object = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown premise kind {self.kind!r}")
        if self.kind in ("probability", "joint"):
            v = self.value
            if isinstance(v, bool) or not isinstance(v, (int, Fraction)) or not 0 <= v <= 1:
                raise ValueError(f"{self.premise_id}: a probability is an exact number in [0, 1], got {v!r}")
            object.__setattr__(self, "value", Fraction(v))


@dataclass(frozen=True)
class Premises:
    items: tuple[Premise, ...]
    latent: frozenset[str] = frozenset()
    assumptions: tuple[str, ...] = ()

    @classmethod
    def of(cls, items: Iterable[Premise], latent: Iterable[str] = (), assumptions: Iterable[str] = ()) -> "Premises":
        ordered = tuple(sorted(set(items), key=lambda p: (p.premise_id, p.kind, str(p.value))))
        return cls(ordered, frozenset(latent), tuple(sorted(set(assumptions))))

    @property
    def conflict(self) -> Optional[str]:
        """The first premise id that carries two different values, else None (ILL_FORMED / CONFLICTING_VALUE)."""
        seen: dict[str, object] = {}
        for p in self.items:
            if p.premise_id in seen and seen[p.premise_id] != p.value:
                return p.premise_id
            seen[p.premise_id] = p.value
        return None

    def get(self, premise_id: str):
        for p in self.items:
            if p.premise_id == premise_id:
                return p.value
        return None

    def ids(self) -> frozenset[str]:
        """Item ids and one latent:V per declared latent variable."""
        return frozenset(p.premise_id for p in self.items) | {f"latent:{v}" for v in self.latent}

    def without(self, premise_id: str) -> "Premises":
        latent = self.latent - {premise_id[len("latent:"):]} if premise_id.startswith("latent:") else self.latent
        return Premises(tuple(p for p in self.items if p.premise_id != premise_id), latent, self.assumptions)


@dataclass(frozen=True)
class Question:
    query_type: str
    polarity: Optional[str] = None
    treatment: str = "X"
    outcome: str = "Y"
    intervention: tuple[tuple[str, int], ...] = ()
    evidence: tuple[tuple[str, int], ...] = ()
    outcome_value: Optional[int] = None
    methods: tuple[tuple[str, ...], ...] = ()
    target_type: Optional[str] = None


@dataclass(frozen=True)
class Outcome:
    status: str  # ANSWER, NOT_COMPUTABLE, ILL_FORMED
    answer: Optional[str] = None
    value: Optional[Fraction] = None
    interval: Optional[tuple[Fraction, Fraction]] = None
    precision_conflict: bool = False
    recipe: Optional[str] = None
    certificate: tuple[str, ...] = ()
    indispensable: tuple[str, ...] = ()
    reason: Optional[str] = None
    missing: tuple[str, ...] = field(default=())
