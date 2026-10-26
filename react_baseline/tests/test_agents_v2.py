"""Intent-mode ReAct, the MVP prompt snapshot, random-intent, and the v2 pilot gate."""
import json
from pathlib import Path

import pytest

from react_baseline import pilot_check, prompts, run as run_cli
from react_baseline.agents import RandomIntentAgent, ReActAgent, parse_intent_response
from react_baseline.llm import MockIntentLLM, MockLLM
from react_baseline.runner import Budgets, run

CFG = {"backend": "mock", "model": "mock"}
SNAP = json.loads((Path(__file__).parent / "snapshots" / "mvp_prompts.json").read_text(encoding="utf-8"))


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# ---------------------------------------------------------------- MVP snapshot
def test_primitive_prompts_match_mvp_snapshot(tmp_path):
    """react-raw sends the MVP system prompt byte-for-byte; its turn prompts use the shared
    translator, so this snapshot was regenerated for the v2.1 event wording (EXPERIMENT_LOG)."""
    assert prompts.system_prompt(1) == SNAP["system_k1"]
    assert prompts.system_prompt(3) == SNAP["system_k3"]
    llm = MockLLM(script=["Thought: a\nAction: 0", "garbage",
                          "Thought: b\nAction: 1\nBug report: odd plate", "Thought: c\nAction: 2"])
    run(ReActAgent(llm, history=10), "V2_task_01_tomato", 0, Budgets(max_env_steps=4), tmp_path / "a", CFG)
    assert [list(c) for c in llm.calls] == SNAP["k1_calls"]
    llm = MockLLM(script=["Thought: go\nAction: 0, 0, 3", "nope", "Thought: back\nAction: 2,2"])
    run(ReActAgent(llm, history=10, actions_per_call=3), "V2_task_01_tomato", 0,
        Budgets(max_env_steps=6), tmp_path / "b", CFG)
    assert [list(c) for c in llm.calls] == SNAP["k3_calls"]


def test_intent_prompt_changes_only_action_section():
    prim, intent = prompts.system_prompt(1), prompts.system_prompt(1, "intent")
    assert prompts.PRIMITIVE_ACTIONS_SECTION in prim and prompts.INTENT_ACTIONS_SECTION in intent
    assert intent.replace(prompts.INTENT_ACTIONS_SECTION, prompts.PRIMITIVE_ACTIONS_SECTION) \
        .replace(prompts.action_spec(1, "intent"), prompts.action_spec(1)) == prim
    rem = prompts.turn_prompt("", "OBS", 1, 1, True, "intent")
    assert "Action: one intent, e.g. interact(0,3)" in rem


# ---------------------------------------------------------------- intent parsing
@pytest.mark.parametrize("text,intents", [
    ("Thought: chop\nAction: interact(0,3)", ["interact(0,3)"]),
    ("**Thought:** x\n**Action:** `goto(3, 1)`", ["goto(3,1)"]),
    ("Thought: x\nAction: wait(4)", ["wait(4)"]),
    ("Thought: x\nAction: left", ["left"]),
    ("Thought: x\nAction: 2", ["left"]),
    ("Thought: x\nAction: interact(sink)", ["interact(sink)"]),
    ("Thought: x\nAction: goto(1,3); interact(0,3)", ["goto(1,3)"]),   # truncated to 1
    ("I will chop. Action: interact(0,3)", ["interact(0,3)"]),
])
def test_parse_intent_response_ok(text, intents):
    p = parse_intent_response(text, 1)
    assert p.parse_error is None
    assert [i.text for i in p.intents] == intents


@pytest.mark.parametrize("text", ["no action here", "Thought: x\nAction: dance", "Thought: x\nAction: wait(99)", ""])
def test_parse_intent_response_errors(text):
    p = parse_intent_response(text, 1)
    assert p.parse_error and p.intents == []


def test_parse_intent_response_multi_and_report():
    p = parse_intent_response("Thought: t\nAction: goto(1,3); interact(0,3); wait\nBug report: board ate it", 2)
    assert [i.text for i in p.intents] == ["goto(1,3)", "interact(0,3)"]
    assert "truncated 3 intents to 2" in p.parse_note and p.bug_report == "board ate it"


# ---------------------------------------------------------------- intent-mode ReAct runs
def test_react_intent_run_history_fallback_and_logs(tmp_path):
    script = ["Thought: get tomato\nAction: interact(0,2)",
              "gibberish",
              "Thought: off map\nAction: goto(9,9)",
              "Thought: board\nAction: interact(cutting board)\nBug report: tomato looks odd"]
    llm = MockLLM(script=script)
    agent = ReActAgent(llm, history=10, action_space="intent")
    s = run(agent, "V2_task_01_tomato", 0, Budgets(max_env_steps=6), tmp_path, CFG)
    steps, calls = _read(tmp_path / "steps.jsonl"), _read(tmp_path / "llm_calls.jsonl")
    assert [r["action"] for r in steps] == [1, 2, 4, 4, 1, 2]
    assert [r["decision_id"] for r in steps] == [0, 0, 1, 2, 3, 3]
    assert [r["intent_step_index"] for r in steps] == [0, 1, 0, 0, 0, 1]
    assert [c["parsed_intents"] for c in calls] == [["interact(0,2)"], ["wait"], ["goto(9,9)"],
                                                    ["interact(cutting board)"]]
    assert calls[2]["executor_result"][0]["error"] == "goto(9,9) failed: outside the map"
    assert calls[0]["executor_result"][0]["stop_reason"] == "completed"
    assert s["agent"] == "react" and s["action_space"] == "intent"
    assert s["parse_errors"] == 1 and s["intent_failures"] == 1 and s["intents_executed"] == 4
    assert s["decisions"] == 4 and s["llm_calls_per_decision"] == 1.0
    assert s["llm_calls_per_env_step"] == pytest.approx(4 / 6)
    # history: intent, executor trace, result; the error and the reminder reach the agent
    u = llm.calls[3][1]
    assert ("[turn 1 | ep 1 | steps 1–2 (2 steps)] Thought: get tomato | Action: interact(0,2) | "
            "Executor: interact(0,2): 2 step(s) [down,left]; events: moved; picked up tomato from the "
            "counter at (0,2); stop: completed | Result: you at (1,2) holding tomato") in u
    assert "[turn 3 | ep 1 | step 4 (1 step)] Thought: off map" in u
    assert "error: goto(9,9) failed: outside the map" in u
    assert "FORMAT REMINDER" in llm.calls[2][1] and "FORMAT REMINDER" not in u
    assert llm.calls[0][0] == prompts.system_prompt(1, "intent")


def test_react_intent_mock_is_deterministic(tmp_path):
    def go(d, seed):
        run(ReActAgent(MockIntentLLM(seed=seed), action_space="intent"), "V2_task_01_tomato", seed,
            Budgets(max_env_steps=150), tmp_path / d, CFG)
        return (tmp_path / d / "steps.jsonl").read_text(), (tmp_path / d / "transitions.jsonl").read_text()
    assert go("a", 0) == go("b", 0)
    assert go("a", 0)[0] != go("c", 1)[0]


def test_random_intent_deterministic_and_uses_reader(tmp_path):
    def go(d, seed, steps=300):
        s = run(RandomIntentAgent(seed), "V2_task_01_tomato", seed, Budgets(max_env_steps=steps),
                tmp_path / d, {"backend": None, "model": None})
        return s, (tmp_path / d / "steps.jsonl").read_text()
    s0, a = go("a", 0)
    _, b = go("b", 0)
    _, c = go("c", 1)
    assert a == b and a != c
    assert s0["agent"] == "random-intent" and s0["llm_calls"] == 0 and s0["action_space"] == "intent"
    assert s0["intent_failure_rate"] == 0.0     # only valid intents are sampled
    rows = [json.loads(line) for line in a.splitlines()]
    assert any(r["intent"].startswith("goto(") for r in rows)
    assert any(r["intent"].startswith("interact(") for r in rows)
    assert s0["decisions"] < s0["total_env_steps"]    # macros take several steps
    # private RNG: episodes differ although the env reseeds the global RNG on reset
    _, long = go("d", 0, 450)
    eps = [[r["intent"] for r in map(json.loads, long.splitlines()) if r["episode"] == e] for e in (1, 2)]
    assert eps[0] and eps[1] and eps[0] != eps[1]


# ---------------------------------------------------------------- CLI + pilot gate
def test_cli_agents_and_pilot_flag(tmp_path, capsys):
    out = tmp_path / "p"
    rc = run_cli.main(["--agent", "random-intent", "--pilot", "--out-dir", str(out), "--quiet"])
    text = capsys.readouterr().out
    d = sorted(out.iterdir())[0]
    s = json.loads((d / "summary.json").read_text())
    assert s["total_env_steps"] == 50
    assert "PILOT CHECK" in text and "longest run of identical intents" in text
    assert rc == (0 if "PILOT PASSED" in text else 1)
    for agent in ("react", "react-raw"):
        assert run_cli.main(["--agent", agent, "--llm", "mock", "--max-env-steps", "30",
                             "--out-dir", str(tmp_path / agent), "--quiet"]) == 0
        cfg = json.loads((sorted((tmp_path / agent).iterdir())[0] / "config.json").read_text())
        assert cfg["action_space"] == ("intent" if agent == "react" else "primitive")
        assert cfg["model_profile"]["name"] == "mock"
    a = run_cli.parse_args(["--agent", "react", "--pilot"])
    assert a.max_env_steps == 50 and Path(a.out_dir) == Path("runs/pilot")


def test_pilot_gate_fails_stuck_intent_agent(tmp_path):
    run(ReActAgent(MockLLM(script=["Thought: bin it\nAction: interact(1,4)"]), action_space="intent"),
        "V2_task_01_tomato", 0, Budgets(max_env_steps=50), tmp_path / "stuck", CFG)
    ok, lines = pilot_check.check(tmp_path / "stuck")
    assert not ok
    assert any(line.startswith("[FAIL] longest run of identical intents:") for line in lines)
    assert not any("same-action streak" in line for line in lines)

    run(ReActAgent(MockLLM(script=["Thought: x\nAction: goto(9,9)"]), action_space="intent"),
        "V2_task_01_tomato", 0, Budgets(max_env_steps=20), tmp_path / "bad", CFG)
    ok, lines = pilot_check.check(tmp_path / "bad")
    assert any(line.startswith("[FAIL] intent failure rate: 100.0%") for line in lines)
