"""Agent-visible text uses the legend / STATIONS display names, never SAGE's internal names
(e.g. "putdown knife succeeded" made the Haiku pilot file two false bug reports)."""
import json
import random
import re

import pytest

from react_baseline.action_reader import ActionReader, VisibleStep, valid_intents
from react_baseline.env_adapter import EnvAdapter, display_name, list_tasks

# internal names that must not appear in agent-visible text (display names in comments)
RAW = [
    re.compile(r"\bknife\b"),                                    # cutting board
    re.compile(r"trash_can"),                                    # trash can
    re.compile(r"ricecooker"),                                   # rice cooker
    re.compile(r"\bblock\b"),                                    # wall
    re.compile(r"\bpan\b(?! \(stove\))"),                        # pan (stove)
    re.compile(r"\bdelivery\b(?! window| ACCEPTED| REJECTED)"),  # delivery window
]


def _raw_hits(text: str) -> list[str]:
    return [p.pattern for p in RAW if p.search(text)]


def test_display_name_mapping():
    assert display_name("knife") == "cutting board" and display_name("trash_can") == "trash can"
    assert display_name("delivery") == "delivery window" and display_name("ricecooker") == "rice cooker"
    assert display_name("pan") == "pan (stove)" and display_name("block") == "wall"
    assert display_name("tomato") == "tomato" and display_name("counter") == "counter"
    # the detector catches the old labels, and describe_event no longer produces them
    assert _raw_hits("putdown knife succeeded") and _raw_hits("nothing happened (bumped trash_can)")
    assert not _raw_hits("putdown cutting board succeeded; delivery REJECTED; pan (stove): empty")
    from react_baseline.env_adapter import describe_event
    info = {"item_log": [{"event_type": "putdown", "target": "knife",
                          "details": {"others": {"success": True}}}]}
    assert describe_event(info) == "putdown cutting board succeeded"
    info = {"item_log": [{"event_type": "None", "target": "trash_can", "details": {"others": {}}}]}
    assert describe_event(info) == "nothing happened (bumped trash can)"


@pytest.mark.parametrize("task", list_tasks())
def test_no_internal_names_in_agent_visible_text(task):
    rng, reader = random.Random(0), ActionReader()
    adapter = EnvAdapter(task)
    texts = []
    try:
        texts.append(adapter.reset())
        state, steps = adapter.state, 0

        def step(a):
            r = adapter.step(a)
            texts.extend([r.obs_text, r.summary, json.dumps(r.state.get("last_event"))])
            return VisibleStep(r.obs_text, r.summary, r.state, r.state["last_event"]["text"], r.done)

        while steps < 250:
            res = reader.execute(rng.choice(valid_intents(state)), state, step)
            steps += res.n_steps
            texts.append(res.trace)
            state = res.final.state
            if res.done:
                texts.append(adapter.reset())
                state = adapter.state
    finally:
        adapter.close()
    bad = [(t[:300], _raw_hits(t)) for t in texts if _raw_hits(t)]
    assert not bad, bad[:3]
