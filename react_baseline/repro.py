"""Agent-driven reproduction: a runner-level protocol, reused unchanged by any agent.

After an agent files a bug report while exploring, the runner may start a
reproduction attempt: reset the env, tell the agent (through optional hooks) to
reproduce *its own report*, and give it ``steps`` env steps. Whether to attempt is
decided only from agent-side information (the report text, attempt counts, steps
used); the oracle is used only afterwards, for scoring.

Agent hooks (all optional; agents without them, e.g. the random baselines, never
report and so never enter an attempt):
    on_repro_start(report: str, steps: int)   on_repro_tick(steps_left: int)   on_repro_end()
An agent signals success by replying with a line "Reproduced: yes"
(``llm_call["parsed_reproduced"]``).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

RESET_NOTE = "the kitchen was reset to its starting state"
START_MARKER = f"=== REPRODUCTION ATTEMPT: {RESET_NOTE} ==="
END_MARKER = f"=== END REPRODUCTION: {RESET_NOTE} ==="
TASK_TEMPLATE = ('REPRODUCTION TASK: reproduce the defect you reported: "{report}". '
                 'Steps left: {steps}. If you believe it happened again, add a line "Reproduced: yes".')
_REPRODUCED = re.compile(r"^[\s*#>_`-]*reproduced[\s*_`]*[:：]\s*[*_`]*\s*(yes|true|y)\b",
                         re.IGNORECASE | re.MULTILINE)


def parse_reproduced(text: str) -> bool:
    return bool(_REPRODUCED.search(text or ""))


def task_block(report: str, steps_left: int) -> str:
    return TASK_TEMPLATE.format(report=report, steps=steps_left)


@dataclass
class ReproConfig:
    enabled: bool = False
    steps: int = 100            # env steps per attempt
    max_attempts: int = 5       # per run
    max_fraction: float = 0.3   # of the run's env-step budget, across all attempts
    dedup_jaccard: float = 0.8  # token Jaccard at or above this = near-duplicate report

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def near_duplicate(a: str, b: str, threshold: float) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if ta == tb:
        return True
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= threshold


@dataclass
class ReproController:
    cfg: ReproConfig
    max_env_steps: int
    attempted_reports: list = field(default_factory=list)
    attempts: list = field(default_factory=list)      # finished attempt rows
    skipped: Counter = field(default_factory=Counter)
    current: dict | None = None
    steps_used: int = 0                               # repro env steps over all attempts

    @property
    def phase(self) -> str:
        return "repro" if self.current is not None else "explore"

    @property
    def attempt_id(self):
        return self.current["attempt_id"] if self.current else None

    def consider(self, report: str, global_step: int) -> str | None:
        """Agent-side decision. Returns None to start an attempt, else the skip reason."""
        if not self.cfg.enabled or self.current is not None:
            return "disabled" if not self.cfg.enabled else "in_attempt"
        if global_step >= self.max_env_steps:
            reason = "no_budget"
        elif any(near_duplicate(report, r, self.cfg.dedup_jaccard) for r in self.attempted_reports):
            reason = "duplicate"
        elif len(self.attempted_reports) >= self.cfg.max_attempts:
            reason = "max_attempts"
        elif self.steps_used + self.cfg.steps > self.cfg.max_fraction * self.max_env_steps:
            reason = "max_fraction"
        else:
            return None
        self.skipped[reason] += 1
        return reason

    def start(self, report: str, source: dict, source_strict: list[int], global_step: int):
        self.attempted_reports.append(report)
        self.current = {
            "attempt_id": len(self.attempts), "source_report": report,
            "source_decision_id": source.get("decision_id"),
            "source_llm_call_id": source.get("llm_call_id"),
            "source_global_step": source.get("global_step"),
            "source_bug_ids_strict": sorted(source_strict),
            "start_global_step": global_step,
            "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0,
            "agent_claimed_reproduced": False,
        }

    def steps_in_attempt(self, global_step: int) -> int:
        return global_step - self.current["start_global_step"] if self.current else 0

    def steps_left(self, global_step: int) -> int:
        return max(self.cfg.steps - self.steps_in_attempt(global_step), 0)

    def finish(self, end_reason: str, global_step: int, step_rows: list[dict]) -> dict:
        """Score with the oracle (post hoc) and close the attempt."""
        cur, self.current = self.current, None
        aid = cur["attempt_id"]
        rows = [r for r in step_rows if r.get("repro_attempt_id") == aid]
        strict = set(cur["source_bug_ids_strict"])
        fired = {b for r in rows for b in r["bug_ids_triggered"]}
        hits = [r["global_step"] for r in rows if strict & set(r["bug_ids_triggered"])]
        cur.update({
            "end_global_step": global_step, "end_reason": end_reason, "steps": len(rows),
            "oracle_reproduced": (bool(strict & fired) if strict else None),
            "reproduced_bug_ids": sorted(strict & fired),
            "first_repro_step": hits[0] if hits else None,
            "other_bug_ids_seen": sorted(fired - strict),
            "usd": round(cur["usd"], 6),
        })
        self.steps_used += len(rows)
        self.attempts.append(cur)
        return cur

    def summary(self) -> dict:
        scored = [a for a in self.attempts if a["oracle_reproduced"] is not None]
        return {
            "repro_attempts": len(self.attempts),
            "repro_skipped": dict(sorted(self.skipped.items())),
            "repro_success_rate": (sum(a["oracle_reproduced"] for a in scored) / len(scored)) if scored else None,
            "unique_bugs_reproduced": len({b for a in self.attempts for b in a["reproduced_bug_ids"]}),
            "unique_bug_ids_reproduced": sorted({b for a in self.attempts for b in a["reproduced_bug_ids"]}),
            "agent_claim_accuracy": (sum(a["agent_claimed_reproduced"] == a["oracle_reproduced"]
                                         for a in scored) / len(scored)) if scored else None,
        }
