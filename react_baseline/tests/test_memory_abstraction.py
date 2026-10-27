"""TKG abstraction: coordinate-free keys from structured agent-visible data."""
import random
import re

import pytest

from memory.abstraction import (abstract, ctx0_key, item_str, outcome_text, predicted_ctx0, situation_text,
                                tile_str)
from react_baseline.action_reader import ActionReader, VisibleStep, parse_intent, reachable_floor, valid_intents
from react_baseline.env_adapter import EnvAdapter

COORD = re.compile(r"\d+\s*,\s*\d+")


def _play(task, intents):
    """Run intents; -> list of (intent, ExecResult, state before)."""
    adapter, reader, out = EnvAdapter(task), ActionReader(), []
    try:
        adapter.reset()
        state = adapter.state

        def step(a):
            r = adapter.step(a)
            return VisibleStep(r.obs_text, r.summary, r.state, r.state["last_event"]["text"], r.done)

        for it in intents:
            intent = parse_intent(it) if isinstance(it, str) else it(state)
            res = reader.execute(intent, state, step)
            out.append((intent, res, state))
            state = res.final.state
            if res.done:
                adapter.reset()
                state = adapter.state
    finally:
        adapter.close()
    return out


def test_entity_strings():
    assert item_str({"name": "plate", "dirty": False, "contents": [{"name": "tomato", "chopped": True}]}) \
        == "plate:clean:[chopped tomato]"
    assert tile_str({"kind": "cutting board", "holding": {"name": "onion", "chopped": False}}) \
        == "cutting board:holding:unchopped onion"
    assert tile_str({"kind": "counter", "holding": None}) == "counter:empty"
    assert tile_str({"kind": "floor", "holding": None}) == "floor" and tile_str(None) == "none"
    assert ctx0_key("goto", None, "floor") == "move | hands:nothing | floor"
    assert item_str({"name": "steak", "chopped": True, "cooked": True}) == "chopped cooked steak"


def test_same_situation_at_two_positions_gives_same_keys():
    # task 01: (1,0) and (2,0) are both empty counters; bump each with empty hands
    (_, a, _), (_, b, _) = _play("V2_task_01_tomato", ["interact(1,0)", "interact(2,0)"])
    ka = abstract(a.kind, a.event_records[-1], None)
    kb = abstract(b.kind, b.event_records[-1], None)
    assert ka["ctx0"] == kb["ctx0"] == "interact | hands:nothing | counter:empty"
    assert ka["outcome"] == kb["outcome"]
    assert a.event_records[-1]["tile"] != b.event_records[-1]["tile"]   # different tiles, same key


@pytest.mark.parametrize("task", ["V2_task_01_tomato", "V2_task_07_tomato_steak",
                                  "V2_task_11_tomato_lettuce_onion"])
def test_keys_have_no_coordinates_and_prediction_matches(task):
    rng = random.Random(1)

    def cheap_like(state):   # the cheap policy's candidate kinds
        me = tuple(state["agent"]["pos"])
        floor = [p for p in reachable_floor(state) if p != me]
        pool = [i for i in valid_intents(state) if i.startswith("interact")] + ["wait(3)"]
        pool += [f"goto({x},{y})" for x, y in rng.sample(floor, min(3, len(floor)))]
        return parse_intent(rng.choice(pool))

    prev = None
    for intent, res, state in _play(task, [cheap_like] * 150):
        k = abstract(res.kind, res.event_records[-1], prev)
        prev = k["outcome"]
        for key in ("ctx0", "ctx1", "outcome", "target"):
            assert not COORD.search(k[key]), k[key]
        if res.stop_reason == "completed":   # (an episode end can cut an intent short)
            assert predicted_ctx0(intent, state) == k["ctx0"], (intent.text, res.trace)
        assert not COORD.search(outcome_text(k["outcome"]) + situation_text(k["ctx0"]))
