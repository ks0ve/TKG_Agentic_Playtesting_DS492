"""No-LLM novelty policy driven by the TKG, and the standalone ``cheap`` agent (RQ1 reference).

Candidates from the visible state: interact(x,y) for every reachable non-floor tile, wait(3), and
goto to 3 random reachable floor tiles. Each scores 1 / (1 + tkg.count(ctx0)), where ctx0 is the
abstract situation the candidate would produce (memory.abstraction.predicted_ctx0); the highest
score wins (ties: seeded random), and with probability epsilon a uniform random candidate is
taken instead. Intents go through the shared Action reader like every other agent's.

``CheapAgent`` builds its TKG from the runner's ``on_transition`` hook (agent-visible data only)
and writes tkg.json; the TKG agent (tkg_agent.py) extends it with the memory check and the LLM.
Ported from the novelty agent on origin/cheap-agent-baseline (tile-visit counts -> TKG counts,
raw buttons -> intents).
"""
from __future__ import annotations

import random

from memory.abstraction import abstract, predicted_ctx0
from memory.tkg import TKG

from .action_reader import Intent, parse_intent, reachable_floor, valid_intents
from .agents import Decision


class CheapPolicy:
    def __init__(self, rng: random.Random, epsilon: float = 0.2, n_goto: int = 3):
        self.rng, self.epsilon, self.n_goto = rng, epsilon, n_goto

    def candidates(self, state: dict) -> list[Intent]:
        out = [parse_intent(i) for i in valid_intents(state) if i.startswith("interact(")]
        out.append(parse_intent("wait(3)"))
        me = tuple(state["agent"]["pos"])
        floor = [p for p in reachable_floor(state) if p != me]
        out += [parse_intent(f"goto({x},{y})") for x, y in self.rng.sample(floor, min(self.n_goto, len(floor)))]
        return out

    def choose(self, state: dict, tkg: TKG) -> Intent:
        cands = self.candidates(state)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(cands)
        scores = [1.0 / (1 + tkg.count(predicted_ctx0(c, state))) for c in cands]
        best = max(scores)
        return self.rng.choice([c for c, s in zip(cands, scores) if s == best])


def build_transition(r, info: dict, prev_outcome: str | None, actor: str) -> dict:
    """A Transition node for one executed intent (ExecResult ``r``), from agent-visible data."""
    ev = r.event_records[-1] if r.event_records else None
    episode_key = f"{info['run']}#ep{info['episode']}"
    return {"id": f"t{info['global_step_start']}", "run": info["run"], "episode": info["episode"],
            "episode_key": episode_key, "global_step_start": info["global_step_start"],
            "global_step_end": info["global_step_end"], "intent": r.intent, "phase": info.get("phase"),
            **abstract(r.kind, ev, prev_outcome), "event_text": (ev or {}).get("text"),
            "actor": actor, "deviation": None, "escalated": False}


class CheapAgent:
    """The cheap policy alone: no LLM, TKG novelty counts, intents through the Action reader."""

    name = "cheap"
    action_space = "intent"

    def __init__(self, seed: int = 0, epsilon: float = 0.2, n_goto: int = 3, min_support: int = 3):
        self.rng = random.Random(seed)   # private: the env reseeds the global RNG on reset
        self.policy = CheapPolicy(self.rng, epsilon, n_goto)
        self.tkg = TKG(min_support)
        self.env_steps = 0
        self.last_transition: dict | None = None
        self._actor = "cheap"            # who chose the decision being executed

    def act(self, obs_text: str, state: dict | None = None) -> Decision:
        self._actor = "cheap"
        return Decision(intents=[self.policy.choose(state, self.tkg)])

    def observe(self, executed, summary, results=None):
        pass

    def on_episode_reset(self):
        pass

    # -- memory ---------------------------------------------------------------------
    def assess(self, t: dict):
        """Memory check before ``t`` is added: -> (deviation dict | None, expectation id | None)."""
        return None, None

    def on_transition(self, r, prev_state, new_state, info) -> dict:
        ek = f"{info['run']}#ep{info['episode']}"
        t = build_transition(r, info, self.tkg.prev_outcome(ek), self._actor)
        t["deviation"], assessed = self.assess(t)
        self.tkg.add(t, prev_state, new_state, assessed)
        self.env_steps += r.n_steps
        self.last_transition = self.tkg.transitions[t["id"]]
        return {"actor": t["actor"], "deviation": t["deviation"]}

    def summary_fields(self) -> dict:
        return {"tkg_nodes": self.tkg.n_nodes, "tkg_edges": len(self.tkg.edges),
                "tkg_transitions": len(self.tkg.transitions)}

    def write_artifacts(self, run_dir):
        self.tkg.export_json(run_dir / "tkg.json")
