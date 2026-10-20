"""Structured, agent-visible transitions: one per executed intent.

``prev_state`` / ``new_state`` are ``env_adapter.visible_state`` dicts, so a
transition holds only what the agent could see. Written to transitions.jsonl;
the future TKG memory consumes these instead of re-parsing observation text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class VisibleTransition:
    prev_state: dict
    intent: str
    primitive_actions: list[int]
    events: list[str]
    new_state: dict
    # bookkeeping (agent-visible or harness counters; no oracle fields)
    decision_id: int = 0
    intent_id: int = 0
    episode: int = 1
    global_step_start: int = 0
    global_step_end: int = 0
    kind: str = ""
    event_records: list = field(default_factory=list)
    stop_reason: str = ""
    error: str | None = None
    note: str | None = None
    episode_ended: bool = False

    def to_row(self) -> dict:
        return asdict(self)
