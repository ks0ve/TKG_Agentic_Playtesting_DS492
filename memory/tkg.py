"""In-memory temporal knowledge graph of executed intents, with a JSON export.

Transition-centric schema:

    (Transition) -NEXT-> (Transition)              consecutive transitions within one episode
    (Transition) -FROM-> (State)  -TO-> (State)    exact visible states before / after
    (Transition) -TARGET-> (Entity)                the bumped / entered tile's entity state
    (Transition) -ASSESSED_AGAINST-> (Expectation) the expectation the memory check used
    (State) -HAS_ENTITY_STATE-> (Entity)

One transition = one executed intent. State ids hash the visible state without episode_step and
last_event. Entities are coordinate-free entity-state strings, shared between states.
Expectations are keyed on the abstraction (ctx1 and ctx0, see memory.abstraction), never on exact
states, so they generalise: each holds an outcome -> count table. Only the observed source exists
for now (an LLM-drafted reference source comes later).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .abstraction import hands_str, item_str, tile_str

SCHEMA = {
    "nodes": ["Transition", "State", "Entity", "Expectation"],
    "edges": ["NEXT", "FROM", "TO", "TARGET", "ASSESSED_AGAINST", "HAS_ENTITY_STATE"],
}


def state_id(state: dict) -> str:
    core = {k: v for k, v in state.items() if k not in ("episode_step", "last_event")}
    return "s:" + hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def state_entities(state: dict) -> list[str]:
    """Entity-state strings of a visible state: your hands, every station, every counter item."""
    ents = {hands_str(state["agent"]["holding"])}
    for st in state.get("stations", []):
        ents.add(tile_str({"kind": st["label"], "holding": st.get("holding")}))
    for it in state.get("items", []):
        if it.get("where") == "counter":
            ents.add(f"counter:holding:{item_str({k: v for k, v in it.items() if k not in ('pos', 'where')})}")
    return sorted(ents)


class Expectation:
    def __init__(self, level: str, key: str):
        self.level, self.key = level, key
        self.outcomes: Counter = Counter()
        self.first_seen_step = self.last_seen_step = None
        self.source = "observed"

    @property
    def id(self) -> str:
        return f"exp:{self.level}:{self.key}"

    @property
    def support(self) -> int:
        return sum(self.outcomes.values())

    def distribution(self) -> dict[str, float]:
        n = self.support
        return {o: c / n for o, c in self.outcomes.items()} if n else {}

    def to_node(self) -> dict:
        return {"id": self.id, "type": "Expectation", "level": self.level, "key": self.key,
                "outcomes": dict(sorted(self.outcomes.items())), "support": self.support,
                "first_seen_step": self.first_seen_step, "last_seen_step": self.last_seen_step,
                "source": self.source}


class TKG:
    def __init__(self, min_support: int = 3):
        self.min_support = min_support
        self.transitions: dict[str, dict] = {}
        self.states: dict[str, dict] = {}
        self.entities: dict[str, dict] = {}
        self.expectations: dict[str, Expectation] = {}
        self.edges: list[dict] = []
        self._episodes: dict[str, list[str]] = defaultdict(list)

    # -- lookups ------------------------------------------------------------------
    def _exp(self, level: str, key: str, create: bool = False) -> Expectation | None:
        eid = f"exp:{level}:{key}"
        if eid not in self.expectations and create:
            self.expectations[eid] = Expectation(level, key)
        return self.expectations.get(eid)

    def count(self, ctx0: str) -> int:
        e = self._exp("ctx0", ctx0)
        return e.support if e else 0

    def expect(self, ctx1: str | None, ctx0: str, min_support: int | None = None):
        """-> (outcome distribution, level, support, expectation id). Uses ctx1 when it has at
        least ``min_support`` observations, otherwise backs off to ctx0 (``ctx1=None``: ctx0 only)."""
        ms = self.min_support if min_support is None else min_support
        e1 = self._exp("ctx1", ctx1) if ctx1 is not None else None
        if e1 is not None and e1.support >= ms:
            return e1.distribution(), "ctx1", e1.support, e1.id
        e0 = self._exp("ctx0", ctx0)
        return (e0.distribution() if e0 else {}), "ctx0", (e0.support if e0 else 0), f"exp:ctx0:{ctx0}"

    def prev_outcome(self, episode_key: str) -> str | None:
        ids = self._episodes.get(episode_key)
        return self.transitions[ids[-1]]["outcome"] if ids else None

    def episode_path(self, episode_key: str, upto: str | None = None) -> list[dict]:
        """Transitions of one episode from its reset up to (and including) transition ``upto``."""
        out = []
        for tid in self._episodes.get(episode_key, []):
            out.append(self.transitions[tid])
            if tid == upto:
                break
        return out

    # -- building -------------------------------------------------------------------
    def _add_state(self, state: dict) -> str:
        sid = state_id(state)
        if sid not in self.states:
            ents = state_entities(state)
            self.states[sid] = {"id": sid, "type": "State", "n_entities": len(ents)}
            for ent in ents:
                self._add_entity(ent)
                self.edges.append({"type": "HAS_ENTITY_STATE", "source": sid, "target": f"ent:{ent}"})
        return sid

    def _add_entity(self, ent: str) -> str:
        eid = f"ent:{ent}"
        if eid not in self.entities:
            self.entities[eid] = {"id": eid, "type": "Entity", "state": ent}
        return eid

    def add(self, t: dict, from_state: dict, to_state: dict, assessed_against: str | None = None):
        """Add one Transition node (a dict with at least id, episode_key, ctx0, ctx1, outcome,
        target, global_step_start/end) and update both expectation tables."""
        t = dict(t, type="Transition")
        tid = t["id"]
        prev = self._episodes[t["episode_key"]]
        if prev:
            self.edges.append({"type": "NEXT", "source": prev[-1], "target": tid})
        prev.append(tid)
        self.transitions[tid] = t
        self.edges.append({"type": "FROM", "source": tid, "target": self._add_state(from_state)})
        self.edges.append({"type": "TO", "source": tid, "target": self._add_state(to_state)})
        if t["target"] != "none":
            self.edges.append({"type": "TARGET", "source": tid, "target": self._add_entity(t["target"])})
        if assessed_against:
            self.edges.append({"type": "ASSESSED_AGAINST", "source": tid, "target": assessed_against})
        for level in ("ctx1", "ctx0"):
            e = self._exp(level, t[level], create=True)
            e.outcomes[t["outcome"]] += 1
            step = t["global_step_end"]
            e.first_seen_step = step if e.first_seen_step is None else e.first_seen_step
            e.last_seen_step = step
        return t

    # -- export -------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.transitions) + len(self.states) + len(self.entities) + len(self.expectations)

    def export(self) -> dict:
        # an ASSESSED_AGAINST edge may point to an expectation that did not exist yet at check time
        # (a novel situation); it exists after add(), so every edge endpoint is a node
        nodes = (list(self.transitions.values()) + list(self.states.values())
                 + list(self.entities.values()) + [e.to_node() for e in self.expectations.values()])
        return {"schema": SCHEMA, "n_nodes": len(nodes), "n_edges": len(self.edges),
                "nodes": nodes, "edges": self.edges}

    def export_json(self, path: Path | str):
        Path(path).write_text(json.dumps(self.export(), indent=1, sort_keys=True), encoding="utf-8")
