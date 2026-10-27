"""Memory check: does a transition deviate from what memory expects?

``Trigger.check(transition, tkg) -> (fire, reason)``, called *before* the transition is added.
After a check, ``trigger.detail`` describes it (level, support, usual outcome, expectation id) for
logging and the LLM's MEMORY block; it is built only from the TKG.

  TKGTrigger(order=1)   novel: ctx0 never seen; surprising: P(outcome) < tau at >= min_support,
                        under ctx1 (backing off to ctx0 when ctx1 has too little support)
  TKGTrigger(order=0)   the same on ctx0 only (current situation, no temporal context)
  TKGTrigger(order=1, coarse=True)  "tkg1c": lookup ctx1 -> ctx0 -> coarse by min_support; novel only
                        when the coarse (intent kind, target kind) was never seen; at the coarse level
                        the (event type, success) outcome is surprising if its share is < tau
  RandomTrigger(p)      fires with probability p (seeded)
  EveryNTrigger(n)      fires on every n-th check
"""
from __future__ import annotations

import random

from .abstraction import coarse_text, outcome_coarse_text, outcome_text


class Trigger:
    name = "base"

    def __init__(self, min_support: int = 3):
        self.detail: dict = {}
        self.min_support = min_support

    def check(self, transition: dict, tkg) -> tuple[bool, str | None]:
        raise NotImplementedError

    def _memory_detail(self, transition: dict, tkg, ctx1) -> dict:
        dist, level, support, eid = tkg.expect(ctx1, transition["ctx0"], self.min_support)
        usual = max(sorted(dist), key=lambda o: dist[o]) if dist else None
        e = tkg.expectations.get(eid)
        return {"level": level, "support": support, "expectation": eid,
                "p_outcome": dist.get(transition["outcome"], 0.0), "usual": usual,
                "usual_count": e.outcomes[usual] if (e is not None and usual) else 0,
                "usual_text": outcome_text(usual) if usual else None,
                "ctx0_support": tkg.count(transition["ctx0"])}

    def _coarse_detail(self, transition: dict, tkg) -> dict | None:
        """Coarse-level detail when ctx1 and ctx0 both have < min_support (None otherwise)."""
        d = self._memory_detail(transition, tkg, transition["ctx1"])
        if d["support"] >= self.min_support:
            return d
        e = tkg.coarse_expectation(transition.get("coarse", ""))
        if e is None or e.support == 0:
            return {**d, "level": "coarse", "support": 0, "coarse_support": 0}
        dist = e.distribution()
        usual = max(sorted(dist), key=lambda o: dist[o])
        return {**d, "level": "coarse", "support": e.support, "coarse_support": e.support,
                "expectation": e.id, "p_outcome": dist.get(transition["outcome_coarse"], 0.0),
                "usual": usual, "usual_count": e.outcomes[usual], "usual_text": outcome_coarse_text(usual),
                "situation_text": coarse_text(transition["coarse"])}


class TKGTrigger(Trigger):
    def __init__(self, order: int = 1, min_support: int = 3, tau: float = 0.15, coarse: bool = False):
        super().__init__(min_support)
        self.order, self.tau, self.coarse = order, tau, coarse
        self.name = f"tkg{order}" + ("c" if coarse else "")

    def check(self, transition, tkg):
        if self.coarse:
            d = self.detail = self._coarse_detail(transition, tkg)
            if d["level"] != "coarse":
                return (True, "surprising") if d["p_outcome"] < self.tau else (False, None)
            if d["coarse_support"] == 0:
                return True, "novel"
            return (True, "surprising") if d["p_outcome"] < self.tau else (False, None)
        d = self._memory_detail(transition, tkg, transition["ctx1"] if self.order >= 1 else None)
        self.detail = d
        if d["ctx0_support"] == 0:
            return True, "novel"
        if d["support"] >= self.min_support and d["p_outcome"] < self.tau:
            return True, "surprising"
        return False, None


class RandomTrigger(Trigger):
    name = "random"

    def __init__(self, p: float = 0.1, seed: int = 0):
        super().__init__()
        self.p, self.rng = p, random.Random(seed)

    def check(self, transition, tkg):
        self.detail = self._memory_detail(transition, tkg, transition["ctx1"])
        return (True, "random") if self.rng.random() < self.p else (False, None)


class EveryNTrigger(Trigger):
    name = "every_n"

    def __init__(self, n: int = 10):
        super().__init__()
        self.n, self.i = max(int(n), 1), 0

    def check(self, transition, tkg):
        self.detail = self._memory_detail(transition, tkg, transition["ctx1"])
        self.i += 1
        return (True, "every_n") if self.i % self.n == 0 else (False, None)


def make_trigger(kind: str, *, p: float = 0.1, n: int = 10, tau: float = 0.15, min_support: int = 3,
                 seed: int = 0) -> Trigger:
    if kind == "tkg1":
        return TKGTrigger(1, min_support, tau)
    if kind == "tkg0":
        return TKGTrigger(0, min_support, tau)
    if kind == "tkg1c":
        return TKGTrigger(1, min_support, tau, coarse=True)
    if kind == "random":
        return RandomTrigger(p, seed)
    if kind == "every_n":
        return EveryNTrigger(n)
    raise ValueError(f"unknown trigger {kind!r}")
