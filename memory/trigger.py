"""Memory check: does a transition deviate from what memory expects?

``Trigger.check(transition, tkg) -> (fire, reason)``, called *before* the transition is added.
After a check, ``trigger.detail`` describes it (level, support, usual outcome, expectation id) for
logging and the LLM's MEMORY block; it is built only from the TKG.

  TKGTrigger(order=1)   novel: ctx0 never seen; surprising: P(outcome) < tau at >= min_support,
                        under ctx1 (backing off to ctx0 when ctx1 has too little support)
  TKGTrigger(order=0)   the same on ctx0 only (current situation, no temporal context)
  RandomTrigger(p)      fires with probability p (seeded)
  EveryNTrigger(n)      fires on every n-th check
"""
from __future__ import annotations

import random


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
                "ctx0_support": tkg.count(transition["ctx0"])}


class TKGTrigger(Trigger):
    def __init__(self, order: int = 1, min_support: int = 3, tau: float = 0.15):
        super().__init__(min_support)
        self.order, self.tau = order, tau
        self.name = f"tkg{order}"

    def check(self, transition, tkg):
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
    if kind == "random":
        return RandomTrigger(p, seed)
    if kind == "every_n":
        return EveryNTrigger(n)
    raise ValueError(f"unknown trigger {kind!r}")
