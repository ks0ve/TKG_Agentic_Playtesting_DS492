"""Action reader: parsing, BFS (incl. the env's coordinate transposition), stop
rules, invalid intents, and hand-written intent solutions for every verified demo."""
import json

import pytest

from react_baseline.action_reader import (ActionReader, VisibleStep, bfs_path, parse_intent,
                                          parse_intents, resolve_name, valid_intents)
from react_baseline.agents import RandomAgent
from react_baseline.env_adapter import EnvAdapter, SAGE_SCENARIO_DIR
from react_baseline.runner import Budgets, run

DEMOS = {d["task"]: d for d in json.loads(
    (SAGE_SCENARIO_DIR / "verified_demonstrations.json").read_text(encoding="utf-8-sig"))}


def _step_fn(adapter, log=None, budget=None):
    n = [0]

    def step(a):
        if budget is not None and n[0] >= budget:
            return None
        n[0] += 1
        r = adapter.step(a)
        if log is not None:
            log.append(a)
        return VisibleStep(r.obs_text, r.summary, r.state, r.state["last_event"]["text"], r.done)
    return step


def _run_intents(task, intents, reader=None):
    reader = reader or ActionReader()
    adapter = EnvAdapter(task)
    try:
        adapter.reset()
        state, results = adapter.state, []
        for it in intents:
            r = reader.execute(it, state, _step_fn(adapter))
            results.append(r)
            state = r.final.state
            if r.done:
                break
        return results, adapter
    finally:
        adapter.close()


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("text,canon", [
    ("right", "right"), ("UP", "up"), ("0", "right"), ("3", "up"), ("4", "wait"), ("stay", "wait"),
    ("wait", "wait"), ("wait(5)", "wait(5)"), ("wait 3", "wait(3)"), ("**wait(2)**", "wait(2)"),
    ("goto(3,0)", "goto(3,0)"), ("go to (3, 0)", "goto(3,0)"), ("goto 2,1", "goto(2,1)"),
    ("`goto(1,3)`", "goto(1,3)"), ("move_to(1,3)", "goto(1,3)"), ("[goto(1, 3)]", "goto(1,3)"),
    ("interact(0,3)", "interact(0,3)"), ("Interact (0, 3) to chop", "interact(0,3)"),
    ("use(4,3)", "use(4,3)".replace("use", "interact")), ("interact with 0,3", "interact(0,3)"),
    ("interact(sink)", "interact(sink)"), ("interact(cutting_board)", "interact(cutting board)"),
    ("interact with the sink", "interact(sink)"), ("left (towards tomato)", "left"),
])
def test_parse_intent_ok(text, canon):
    it = parse_intent(text)
    assert it.kind != "invalid", it.error
    assert it.text == canon


@pytest.mark.parametrize("text", ["", "jump", "wait(25)", "wait(0)", "5", "12", "pick up tomato"])
def test_parse_intent_invalid(text):
    it = parse_intent(text)
    assert it.kind == "invalid" and it.error


def test_parse_intents_split_and_truncate():
    its, note = parse_intents("goto(1,3); interact(0,3) then wait(2)", max_n=5)
    assert [i.text for i in its] == ["goto(1,3)", "interact(0,3)", "wait(2)"] and note is None
    its, note = parse_intents("goto(1,3); interact(0,3)", max_n=1)
    assert [i.text for i in its] == ["goto(1,3)"] and "truncated 2 intents to 1" in note


# ---------------------------------------------------------------- BFS + transposition
def _initial_state(task):
    a = EnvAdapter(task)
    try:
        a.reset()
        return a.state
    finally:
        a.close()


def test_bfs_tie_break_and_screen_frame():
    s = _initial_state("V2_task_01_tomato")   # agent at screen (1,1)
    assert s["agent"]["pos"] == [1, 1]
    # (3,3) is 4 moves away; right is expanded before down, so the path goes right first
    path, goal = bfs_path(s, (1, 1), {(3, 3)})
    assert goal == (3, 3) and path == [0, 0, 1, 1]
    path, _ = bfs_path(s, (1, 1), {(1, 3)})
    assert path == [1, 1]
    # the delivery window at (2,2) is not floor: the route to (3,2) avoids it
    path, _ = bfs_path(s, (1, 1), {(3, 2)})
    assert path == [0, 0, 1]
    assert bfs_path(s, (1, 1), {(2, 2)}) == (None, None)


@pytest.mark.parametrize("task", ["V2_task_01_tomato", "V2_task_04_steak", "V2_task_78_onion_fish_rice"])
def test_goto_every_reachable_tile_lands_in_screen_frame(task):
    """goto(x,y) ends with the agent at screen (x,y), i.e. env item.y == x, item.x == y."""
    s0 = _initial_state(task)
    targets = [i for i in valid_intents(s0) if i.startswith("goto(")]
    assert targets
    reader = ActionReader()
    for t in targets:
        adapter = EnvAdapter(task)
        try:
            adapter.reset()
            r = reader.execute(t, adapter.state, _step_fn(adapter))
            x, y = map(int, t[5:-1].split(","))
            assert r.stop_reason == "completed" and r.error is None, r.trace
            assert r.final.state["agent"]["pos"] == [x, y]
            ag = adapter.env.unwrapped.agent[0]
            assert (int(ag.y), int(ag.x)) == (x, y)   # env stores (row, col) transposed
            assert f"YOU: at ({x},{y})" in r.final.obs_text
        finally:
            adapter.close()


def test_interact_walks_adjacent_then_bumps_once():
    results, _ = _run_intents("V2_task_01_tomato", ["interact(0,2)"])
    r = results[0]
    assert r.primitive_actions == [1, 2]  # down to (1,2), then left into the tomato
    assert r.events[-1] == "pickup tomato" and r.stop_reason == "completed"
    assert r.final.state["agent"]["holding"]["name"] == "tomato"


def test_resolve_name():
    s = _initial_state("V2_task_01_tomato")
    assert resolve_name(s, "sink") == ((4, 3), None)
    assert resolve_name(s, "cutting board") == ((0, 3), None)
    assert resolve_name(s, "tomato") == ((0, 2), None)
    pos, err = resolve_name(s, "rice")
    assert pos is None and "no visible" in err


# ---------------------------------------------------------------- failures and stop rules
@pytest.mark.parametrize("intent,msg", [
    ("goto(9,9)", "goto(9,9) failed: outside the map"),
    ("goto(0,0)", "goto(0,0) failed: not a floor tile"),
    ("goto(2,2)", "goto(2,2) failed: not a floor tile"),
    ("interact(2,1)", "interact(2,1) failed: (2,1) is a floor tile; use goto"),
    ("interact(0,0)", "interact(0,0) failed: no floor tile next to it is reachable"),
    ("interact(pizza)", "interact(pizza) failed: no visible station or item called 'pizza'"),
    ("dance", "failed: unparseable intent"),
])
def test_invalid_intent_runs_one_wait_with_error(intent, msg):
    results, _ = _run_intents("V2_task_01_tomato", [intent])
    r = results[0]
    assert r.primitive_actions == [4]
    assert r.stop_reason == "invalid"
    assert msg in r.error
    assert msg in r.trace


def test_goto_current_tile_waits_one_step():
    r = _run_intents("V2_task_01_tomato", ["goto(1,1)"])[0][0]
    assert r.primitive_actions == [4] and r.error is None and "already there" in r.note


def test_length_cap():
    r = _run_intents("V2_task_01_tomato", ["goto(3,3)"], ActionReader(max_macro_len=2))[0][0]
    assert r.primitive_actions == [0, 0] and r.stop_reason == "length_cap"
    r = _run_intents("V2_task_01_tomato", ["wait(20)"], ActionReader(max_macro_len=5))[0][0]
    assert r.primitive_actions == [4] * 5 and r.stop_reason == "length_cap"
    assert ActionReader().max_macro_len == 40


def test_budget_stops_macro():
    adapter = EnvAdapter("V2_task_01_tomato")
    try:
        adapter.reset()
        r = ActionReader().execute("goto(3,3)", adapter.state, _step_fn(adapter, budget=1))
        assert r.primitive_actions == [0] and r.stop_reason == "budget"
    finally:
        adapter.close()


def test_unexpected_bump_stops_goto():
    """The reader plans from the visible state; if a step does not produce the
    expected plain move (here: a stale map claims (3,1) is floor but the env
    bumps), it stops instead of pressing on blindly."""
    s = _initial_state("V2_task_01_tomato")
    stale = json.loads(json.dumps(s))

    calls = []

    def fake_step(a):
        calls.append(a)
        st = json.loads(json.dumps(s))
        if len(calls) == 1:
            st["agent"]["pos"] = [2, 1]
            st["last_event"] = {"type": "move", "target": None, "success": None, "text": "moved"}
        else:  # second move is blocked
            st["agent"]["pos"] = [2, 1]
            st["last_event"] = {"type": None, "target": "plate", "success": None,
                                "text": "nothing happened (bumped plate)"}
        return VisibleStep("obs", "sum", st, st["last_event"]["text"], False)

    r = ActionReader().execute("goto(3,3)", stale, fake_step)
    assert calls == [0, 0]
    assert r.stop_reason == "unexpected_event"
    assert r.events == ["moved", "nothing happened (bumped plate)"]


def test_episode_end_stops_macro():
    seq = ["interact(0,2)", "interact(0,3)", "interact(0,3)", "interact(3,0)", "interact(0,3)",
           "interact(2,2)", "wait(5)"]
    results, _ = _run_intents("V2_task_01_tomato", seq)
    assert results[-1].intent == "interact(2,2)" and results[-1].stop_reason == "episode_end"


# ---------------------------------------------------------------- demos via intents
SOLUTIONS = {
    "V2_task_01_tomato": ["interact(tomato)", "interact(cutting board)", "interact(knife)",
                          "interact(plate)", "interact(0,3)", "interact(delivery window)"],
    "V2_task_02_lettuce": ["interact(0,2)", "interact(4,1)", "interact(4,1)", "interact(3,0)",
                           "interact(4,1)", "interact(2,2)"],
    "V2_task_03_onion": ["interact(4,3)", "interact(3,0)", "interact(3,0)", "interact(3,4)",
                         "interact(3,0)", "interact(2,2)"],
    "V2_task_04_steak": ["interact(1,0)", "interact(2,4)", "interact(2,4)", "interact(2,4)",
                         "interact(pan)", "interact(plate)", "wait(3)", "interact(3,0)",
                         "interact(2,2)"],
    "V2_task_16_fish": ["interact(3,0)", "interact(0,1)", "interact(0,1)", "interact(1,0)",
                        "interact(0,1)", "interact(2,2)"],
    "V2_task_17_rice": ["interact(rice)", "interact(rice cooker)", "interact(plate)", "wait(3)",
                        "interact(2,0)", "interact(delivery)"],
}


@pytest.mark.parametrize("task", sorted(DEMOS))
def test_verified_demo_tasks_solvable_with_intents(task):
    results, _ = _run_intents(task, SOLUTIONS[task])
    traces = "\n".join(r.trace for r in results)
    assert all(r.error is None for r in results), traces
    last = results[-1]
    assert last.done and last.events[-1] == "delivery ACCEPTED (order complete)", traces
    assert len(results) == len(SOLUTIONS[task]), traces


# ---------------------------------------------------------------- runner integration
def test_runner_writes_transitions(tmp_path):
    run(RandomAgent(0), "V2_task_01_tomato", 0, Budgets(max_env_steps=30), tmp_path,
        {"backend": None, "model": None})
    rows = [json.loads(line) for line in (tmp_path / "transitions.jsonl").read_text().splitlines()]
    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == 30 == len(steps)
    for t, s in zip(rows, steps):
        assert t["primitive_actions"] == [s["action"]] and t["intent"] == s["intent"]
        assert t["global_step_start"] == t["global_step_end"] == s["global_step"]
        assert {"prev_state", "new_state", "events", "event_records", "stop_reason"} <= set(t)
    for a, b in zip(rows, rows[1:]):
        if not a["episode_ended"]:
            assert a["new_state"] == b["prev_state"]
