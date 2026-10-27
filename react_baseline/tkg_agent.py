"""TKG arm: cheap policy by default, the LLM only when memory says something deviated.

Phase 1 (the first ``warmup_steps`` env steps): the cheap policy acts and fills the TKG; no memory
check, no LLM. Phase 2, after every executed intent: memory check (``memory.trigger``) against the
TKG, then the transition is added. When the check fires, the next decision goes to the LLM, which
keeps control for up to ``investigate_k`` decisions (refreshed if it fires again) or hands back
early with ``Action: resume``; otherwise the cheap policy acts.

The LLM call is a vanilla ReAct turn (same system prompt, same turn template; the history is the
last 10 transitions, cheap ones marked ``[auto]``) plus one MEMORY block built only from the TKG.
Bug reports go through the runner's reproduction protocol unchanged; reproduction decisions are the
LLM's, and with ``repro_path`` the prompt adds the recorded path: the reported episode's transitions
from the reset up to the report, from the TKG. Nothing here sees the oracle.
"""
from __future__ import annotations

import re
from collections import Counter

from memory.abstraction import outcome_text, situation_text
from memory.trigger import TKGTrigger, Trigger

from .agents import Decision, ReActAgent
from .cheap_policy import CheapAgent

_RESUME = re.compile(r"^[\s*#>_`-]*action[\s*_`]*[:=：]\s*[*_`]*\s*resume\b", re.IGNORECASE | re.MULTILINE)
MAX_PATH_LINES = 40


def memory_block(t: dict, detail: dict, k: int) -> str:
    """MEMORY block for the LLM, from the TKG only (situation, usual outcome, this outcome)."""
    now = outcome_text(t["outcome"])
    situation = situation_text(t["ctx0"])
    if detail.get("usual") and detail.get("ctx0_support"):
        head = (f'MEMORY: In this situation ({situation}) the usual outcome was "{outcome_text(detail["usual"])}" '
                f'(seen {detail["usual_count"]} times). This time: "{now}".')
    else:
        head = f'MEMORY: This situation ({situation}) had not been seen before. This time: "{now}".'
    return (head + f"\nYou have control for up to {k} decisions. Reply with \"Action: resume\" to hand "
            "control back to the automatic explorer.")


def path_block(path: list[dict]) -> str:
    """The recorded path of the reported episode, one compact line per executed intent."""
    lines = [f"{i}. {t['intent']} -> {t.get('event_text') or 'nothing'}" for i, t in enumerate(path, 1)]
    if len(lines) > MAX_PATH_LINES:
        lines = [f"(... {len(lines) - MAX_PATH_LINES} earlier steps)"] + lines[-MAX_PATH_LINES:]
    return ("RECORDED PATH (from your memory: what you did in the reported episode, from the kitchen "
            "reset up to the report):\n" + ("\n".join(lines) or "(no steps before the report)"))


class TKGAgent(CheapAgent):
    name = "tkg"
    action_space = "intent"

    def __init__(self, llm, seed: int = 0, trigger: Trigger | None = None, warmup_steps: int = 200,
                 investigate_k: int = 3, repro_path: bool = True, history: int = 10, min_support: int = 3,
                 warn=print):
        super().__init__(seed, min_support=min_support)
        self.react = ReActAgent(llm, history=history, warn=warn, action_space="intent")
        self.system = self.react.system
        self.trigger = trigger or TKGTrigger(1, min_support)
        self.warmup_steps, self.investigate_k, self.repro_path = warmup_steps, investigate_k, repro_path
        self.investigate_left = 0
        self.pending_reason: str | None = None   # the trigger reason that started / refreshed control
        self.memory: str | None = None           # current MEMORY block
        self.repro_block: str | None = None
        self.escalations = self.checks = self.fires = self.llm_decisions = self.resumes = 0
        self.reasons: Counter = Counter()

    @property
    def in_repro(self) -> bool:
        return self.react.repro_report is not None

    # -- memory check (called by on_transition before the transition is added) ---------
    def assess(self, t: dict):
        if self.env_steps < self.warmup_steps:      # phase 1: fill the TKG only
            return None, None
        fire, reason = self.trigger.check(t, self.tkg)
        d = self.trigger.detail
        deviation = {"fire": fire, "reason": reason, "score": round(1.0 - d.get("p_outcome", 0.0), 4),
                     "p_outcome": round(d.get("p_outcome", 0.0), 4), "level": d.get("level"),
                     "support": d.get("support"), "usual": d.get("usual")}
        if not self.in_repro:
            self.checks += 1
            if fire:
                self.fires += 1
                self.reasons[reason] += 1
                if self.investigate_left == 0:
                    self.escalations += 1
                    t["escalated"] = True
                self.investigate_left = self.investigate_k
                self.pending_reason = reason
                self.memory = memory_block(t, d, self.investigate_k)
        return deviation, d.get("expectation")

    # -- acting -------------------------------------------------------------------------
    def act(self, obs_text: str, state: dict | None = None) -> Decision:
        if self.in_repro:
            reason, block = "repro", self.repro_block
        elif self.investigate_left > 0:
            reason, block = self.pending_reason or "investigate", self.memory
            self.pending_reason = None
            self.investigate_left -= 1
        else:
            return super().act(obs_text, state)
        self.react.extra_block = block
        d = self.react.act(obs_text, state)
        self.react.extra_block = None
        self.llm_decisions += 1
        rec = d.llm_call
        rec["trigger_reason"] = reason
        rec["resumed"] = False
        if reason != "repro" and _RESUME.search(rec["response"] or ""):
            rec.update(resumed=True, parse_error=None, parse_note="resume: control back to the cheap policy")
            self.react.remind_format = False
            self.investigate_left, self.memory, self.resumes = 0, None, self.resumes + 1
            self._actor = "cheap"
            return Decision(llm_call=rec, intents=[self.policy.choose(state, self.tkg)])
        if self.investigate_left == 0 and reason != "repro":
            self.memory = None
        self._actor = "llm"
        return d

    def observe(self, executed, summary, results=None):
        if self._actor == "llm":
            self.react.observe(executed, summary, results)
        else:
            self.react.observe_auto(executed, summary, results)

    # -- episode / reproduction hooks ---------------------------------------------------
    def on_episode_reset(self):
        self.react.on_episode_reset()

    def on_repro_start(self, report: str, steps: int):
        last = self.last_transition
        path = self.tkg.episode_path(last["episode_key"], upto=last["id"]) if last else []
        self.repro_block = path_block(path) if self.repro_path else None
        self.investigate_left, self.memory, self.pending_reason = 0, None, None
        self.react.on_repro_start(report, steps)

    def on_repro_tick(self, steps_left: int):
        self.react.on_repro_tick(steps_left)

    def on_repro_end(self):
        self.repro_block = None
        self.react.on_repro_end()

    def summary_fields(self) -> dict:
        return {**super().summary_fields(),
                "warmup_steps": self.warmup_steps, "trigger": self.trigger.name,
                "investigate_k": self.investigate_k, "repro_path": self.repro_path,
                "escalations": self.escalations, "llm_decisions": self.llm_decisions,
                "llm_resumes": self.resumes, "trigger_checks": self.checks, "trigger_fires": self.fires,
                "trigger_fire_rate": (self.fires / self.checks) if self.checks else None,
                "deviation_reasons": dict(sorted(self.reasons.items()))}
