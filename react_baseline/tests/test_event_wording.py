"""v2.1 event wording (the approved table), the new event-record fields, and history step
ranges that match the observation's 'STEP: k/200' numbering."""
import json
import re

import pytest

from react_baseline.action_reader import ActionReader, VisibleStep
from react_baseline.agents import ReActAgent
from react_baseline.env_adapter import EnvAdapter, describe_event
from react_baseline.llm import MockLLM
from react_baseline.repro import END_MARKER, START_MARKER, ReproConfig
from react_baseline.runner import Budgets, run

CFG = {"backend": "mock", "model": "mock"}
TOMATO = {"name": "tomato", "chopped": False}
CH_TOMATO = {"name": "tomato", "chopped": True}
CH_LETTUCE = {"name": "lettuce", "chopped": True}
ONION = {"name": "onion", "chopped": False}
STEAK_COOKED = {"name": "steak", "chopped": True, "cooked": True}
CH_STEAK = {"name": "steak", "chopped": True, "cooked": False}
RICE = {"name": "rice", "cooked": False}
DOUGH = {"name": "dough"}
SUSHI = {"name": "sushi"}


def plate(*contents, dirty=False):
    return {"name": "plate", "dirty": dirty, "contents": list(contents)}


def tile(kind, holding=None):
    return {"kind": kind, "holding": holding}


def ev(et, target, success=None, *, action=0, hb=None, ha=None, tb=None, ta=None, xy=(3, 0)):
    info = {"item_log": [{"event_type": et, "target": target, "details": {"others": {"success": success}}}]}
    ctx = {"action": action, "held_before": hb, "held_after": ha,
           "tile": None if tb is None else list(xy), "tile_before": tb,
           "tile_after": ta if ta is not None else tb}
    return describe_event(info, ctx)


# (approved row, event, expected text)
TABLE = [
    ("1 wait", ev("None", None, action=4, hb=TOMATO, ha=TOMATO), "waited"),
    ("2 bump wall", ev("None", None, tb=tile("wall")), "nothing happened (bumped wall at (3,0))"),
    ("3 bump counter", ev("None", "counter", tb=tile("counter")),
     "nothing happened (bumped empty counter at (3,0))"),
    ("4 bump station", ev("None", "sink", tb=tile("sink")), "nothing happened (bumped sink at (3,0))"),
    ("5 pickup counter", ev("pickup", "tomato", ha=TOMATO, tb=tile("counter", TOMATO), ta=tile("counter")),
     "picked up tomato from the counter at (3,0)"),
    ("5 pickup board", ev("pickup", "tomato", ha=CH_TOMATO, tb=tile("cutting board", CH_TOMATO),
                          ta=tile("cutting board")),
     "picked up chopped tomato from the cutting board at (3,0)"),
    ("5 take out of pan", ev("pickup", "steak", ha=STEAK_COOKED, tb=tile("pan", STEAK_COOKED), ta=tile("pan")),
     "took chopped cooked steak out of the pan at (3,0)"),
    ("6 pickup plate", ev("pickup", "plate", ha=plate(), tb=tile("counter", plate()), ta=tile("counter")),
     "picked up the empty plate from the counter at (3,0)"),
    ("6 pickup full plate", ev("pickup", "plate", ha=plate(CH_TOMATO), tb=tile("counter", plate(CH_TOMATO)),
                               ta=tile("counter")),
     "picked up the plate with [chopped tomato] from the counter at (3,0)"),
    ("6 pickup dirty plate", ev("pickup", "plate", ha=plate(dirty=True), tb=tile("counter", plate(dirty=True)),
                                ta=tile("counter")),
     "picked up the dirty plate from the counter at (3,0)"),
    ("7 putdown counter", ev("putdown", "counter", True, hb=plate(CH_TOMATO), tb=tile("counter"),
                             ta=tile("counter", plate(CH_TOMATO))),
     "put the plate with [chopped tomato] down on the counter at (3,0)"),
    ("8 putdown onto plate", ev("putdown", "plate", True, hb=CH_LETTUCE, tb=tile("counter", plate(CH_TOMATO)),
                                ta=tile("counter", plate(CH_TOMATO, CH_LETTUCE))),
     "put chopped lettuce onto the plate at (3,0); the plate now holds [chopped tomato, chopped lettuce]"),
    ("8 merge", ev("putdown", "plate", True, hb={"name": "fish", "chopped": True},
                   tb=tile("counter", plate(RICE)), ta=tile("counter", plate(SUSHI))),
     "put chopped fish onto the plate at (3,0); the plate now holds [sushi]"),
    ("9 putdown plate fails", ev("putdown", "plate", False, hb=TOMATO, ha=TOMATO, tb=tile("counter", plate())),
     "tried to put tomato onto the plate at (3,0): nothing changed"),
    ("10 putdown on food", ev("putdown", "onion", False, hb=TOMATO, ha=TOMATO, tb=tile("counter", ONION)),
     "tried to put tomato down at (3,0), where the onion is: nothing changed"),
    ("11 board", ev("putdown", "knife", True, hb=TOMATO, tb=tile("cutting board"), ta=tile("cutting board", TOMATO)),
     "put tomato on the cutting board at (3,0)"),
    ("11 pan", ev("putdown", "pan", True, hb=CH_STEAK, tb=tile("pan"), ta=tile("pan", CH_STEAK)),
     "put chopped steak in the pan at (3,0)"),
    ("11 rice cooker", ev("putdown", "ricecooker", True, hb=RICE, tb=tile("rice cooker"),
                          ta=tile("rice cooker", RICE)),
     "put rice in the rice cooker at (3,0)"),
    ("12 station fails", ev("putdown", "sink", False, hb=TOMATO, ha=TOMATO, tb=tile("sink")),
     "tried to put tomato in the sink at (3,0): nothing changed"),
    ("12 window fails", ev("putdown", "delivery", False, hb=TOMATO, ha=TOMATO, tb=tile("delivery window")),
     "tried to put tomato on the delivery window at (3,0): nothing changed"),
    ("13 sink", ev("putdown", "sink", True, hb=plate(dirty=True), tb=tile("sink"),
                   ta=tile("sink", plate(dirty=True))),
     "put the dirty plate in the sink at (3,0)"),
    ("14 chop", ev("chop", "knife", True, tb=tile("cutting board", TOMATO), ta=tile("cutting board", CH_TOMATO)),
     "chopped tomato on the cutting board at (3,0)"),
    ("15 wash dirty", ev("wash", "sink", tb=tile("sink", plate(dirty=True))),
     "scrubbed the plate in the sink at (3,0) (still dirty)"),
    ("15 wash clean", ev("wash", "sink", tb=tile("sink", plate(dirty=True)), ta=tile("sink", plate())),
     "scrubbed the plate in the sink at (3,0) (now clean)"),
    ("16 add to held plate", ev("pickup_into_plate", "tomato", True, hb=plate(CH_LETTUCE),
                                ha=plate(CH_LETTUCE, CH_TOMATO), tb=tile("cutting board", CH_TOMATO),
                                ta=tile("cutting board")),
     "added chopped tomato from the cutting board at (3,0) to the plate you are holding; "
     "the plate now holds [chopped lettuce, chopped tomato]"),
    ("16 merge", ev("pickup_into_plate", "tomato", True, hb=plate(DOUGH), ha=plate({"name": "pizza"}),
                    tb=tile("counter", CH_TOMATO), ta=tile("counter")),
     "added chopped tomato from the counter at (3,0) to the plate you are holding; the plate now holds [pizza]"),
    ("17 add fails", ev("pickup_into_plate", "tomato", False, hb=plate(), ha=plate(), tb=tile("counter", TOMATO)),
     "tried to add tomato from the counter at (3,0) to the plate you are holding: nothing changed"),
    ("18 empty station", ev("pickup_into_plate", "knife", False, hb=plate(), ha=plate(), tb=tile("cutting board")),
     "tried to add from the empty cutting board at (3,0) to the plate you are holding: nothing changed"),
    ("19 deliver ok", ev("deliver", None, True, hb=plate(CH_TOMATO), tb=tile("delivery window")),
     "delivered the plate with [chopped tomato]: order ACCEPTED"),
    ("20 deliver bad", ev("deliver", None, False, hb=plate(TOMATO), tb=tile("delivery window")),
     "delivered the plate with [tomato]: order REJECTED"),
    ("20 deliver empty", ev("deliver", None, False, hb=plate(), tb=tile("delivery window")),
     "delivered the empty plate: order REJECTED"),
    ("21 dump food", ev("dump", ["tomato"], hb=TOMATO, tb=tile("trash can")),
     "threw away tomato in the trash can at (3,0)"),
    ("22 dump plate", ev("dump", ["plate", "dough"], hb=plate(DOUGH), tb=tile("trash can")),
     "threw away the plate with [dough] in the trash can at (3,0)"),
    ("23 move", ev("move", [1, 2], tb=tile("floor")), "moved"),
    # outcomes are reported, never explained: a 'failed' attempt that changed something says what changed
    ("failed but changed", ev("putdown", "sink", False, hb=TOMATO, ha=None, tb=tile("sink"),
                              ta=tile("sink", TOMATO)),
     "tried to put tomato in the sink at (3,0): you now hold nothing; the sink at (3,0) now holds tomato"),
    ("bump while cooking", ev("None", "pan", tb=tile("pan", CH_STEAK), ta=tile("pan", STEAK_COOKED)),
     "bumped pan at (3,0); the pan at (3,0) now holds chopped cooked steak"),
]


@pytest.mark.parametrize("row,got,want", TABLE, ids=[t[0] for t in TABLE])
def test_wording_table(row, got, want):
    assert got == want


def test_event_record_new_fields_from_the_game():
    """held_before / held_after / tile / tile_before / tile_after come from the real game,
    and the old fields keep their meaning (type, target, success unchanged)."""
    adapter = EnvAdapter("V2_task_01_tomato")
    reader, recs = ActionReader(), []
    try:
        adapter.reset()
        state = adapter.state

        def step(a):
            r = adapter.step(a)
            recs.append(r.state["last_event"])
            return VisibleStep(r.obs_text, r.summary, r.state, r.state["last_event"]["text"], r.done)

        for it in ["interact(0,2)", "interact(0,3)", "interact(0,3)", "interact(0,3)"]:
            state = reader.execute(it, state, step).final.state
    finally:
        adapter.close()
    pick = next(r for r in recs if r["type"] == "pickup")
    assert pick["target"] == "tomato" and pick["success"] is None
    assert pick["held_before"] is None and pick["held_after"] == {"name": "tomato", "chopped": False}
    assert pick["tile"] == [0, 2] and pick["tile_before"] == {"kind": "counter", "holding": pick["held_after"]}
    assert pick["tile_after"] == {"kind": "counter", "holding": None}
    put = next(r for r in recs if r["type"] == "putdown")
    assert put["target"] == "cutting board" and put["success"] is True and put["tile"] == [0, 3]
    assert put["held_after"] is None and put["tile_after"]["holding"]["name"] == "tomato"
    assert put["text"] == "put tomato on the cutting board at (0,3)"
    move = recs[0]
    assert move["type"] == "move" and move["tile_before"] == {"kind": "floor", "holding": None}
    assert set(move) >= {"type", "target", "success", "text", "held_before", "held_after", "tile",
                         "tile_before", "tile_after"}


def test_transitions_carry_the_new_fields(tmp_path):
    run(ReActAgent(MockLLM(script=["Thought: t\nAction: interact(0,2)"]), action_space="intent"),
        "V2_task_01_tomato", 0, Budgets(max_env_steps=3), tmp_path, CFG)
    rows = [json.loads(line) for line in (tmp_path / "transitions.jsonl").read_text().splitlines()]
    for rec in (r for t in rows for r in t["event_records"]):
        assert {"held_before", "held_after", "tile", "tile_before"} <= set(rec)


_LABEL = re.compile(r"\[turn (\d+) \| ep (\d+) \| (?:step (\d+)|steps (\d+)–(\d+)) \((\d+) steps?\)\]")


def test_history_step_ranges_match_observation_steps(tmp_path):
    """Across macros, a natural episode end (200 steps), and a reproduction attempt, every
    history label shows the episode steps of that turn exactly as steps.jsonl records them,
    numbered like the observation's 'STEP: k/200' line, with the observation's episode number."""
    script = ["Thought: a\nAction: interact(0,2)",
              "Thought: b\nAction: wait(20)\nBug report: the tomato vanished",
              "Thought: c\nAction: goto(2,1)",
              "Thought: d\nAction: wait(20)"]
    llm = MockLLM(script=script)
    agent = ReActAgent(llm, history=10, action_space="intent")
    run(agent, "V2_task_01_tomato", 0, Budgets(max_env_steps=300), tmp_path, CFG,
        repro=ReproConfig(enabled=True, steps=10, max_attempts=1, max_fraction=0.5))
    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    by_dec = {}
    for r in steps:
        by_dec.setdefault(r["decision_id"], []).append(r)
    prompts = [u for _, u in llm.calls]
    checked = set()
    for u in prompts:
        for m in _LABEL.finditer(u):
            turn, ep = int(m.group(1)), int(m.group(2))
            first = int(m.group(3) or m.group(4))
            last = int(m.group(3) or m.group(5))
            n = int(m.group(6))
            rows = by_dec[turn - 1]
            assert [r["episode_step"] for r in rows] == list(range(first, last + 1)), (turn, m.group(0))
            assert n == len(rows)
            seen = prompts[turn - 1]   # the observation this turn's decision was made on
            assert f"(episode {ep}):" in seen
            assert int(re.search(r"STEP: (\d+)/200", seen).group(1)) + 1 == first
            checked.add(turn)
    assert len(checked) == len(by_dec) - 1   # every turn but the last was shown
    phases = {r["phase"] for r in steps}
    assert phases == {"explore", "repro"} and any(r["episode_step"] == 200 for r in steps)
    assert any(START_MARKER in u for u in prompts) and any(END_MARKER in u for u in prompts)
    assert any(ReActAgent.RESET_MARKER in u for u in prompts)
    assert START_MARKER == "=== REPRODUCTION ATTEMPT: the kitchen was reset to its starting state ==="
    assert END_MARKER == "=== END REPRODUCTION: the kitchen was reset to its starting state ==="
