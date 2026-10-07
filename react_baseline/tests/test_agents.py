import json

import pytest

from react_baseline.agents import (ContextOverflowError, ReActAgent, parse_response)
from react_baseline.llm import MockLLM
from react_baseline.prompts import FORMAT_REMINDER
from react_baseline.runner import Budgets, run

CFG = {"backend": "mock", "model": "mock"}


@pytest.mark.parametrize("text,k,actions", [
    ("Thought: go right\nAction: 0", 1, [0]),
    ("thought: hm\naction: 3", 1, [3]),
    ("**Thought:** x\n**Action:** 2", 1, [2]),
    ("Thought: x\nAction: 2 (left)", 1, [2]),
    ("Thought: x\nAction: `1`", 1, [1]),
    ("Thought: x\nAction: [4]", 1, [4]),
    ("Thought: x\nAction: left", 1, [2]),
    ("Thought: x\nAction: 0, 0, 1", 3, [0, 0, 1]),
    ("Thought: x\nAction: 0,1,2,3", 2, [0, 1]),   # truncated to actions_per_call
    ("I will move. Action: 1", 1, [1]),           # no line start, still found
])
def test_parse_ok(text, k, actions):
    p = parse_response(text, k)
    assert p.parse_error is None
    assert p.actions == actions


@pytest.mark.parametrize("text", [
    "I think I should go right.",
    "Thought: x\nAction: 5",
    "Thought: x\nAction: 12",
    "Thought: x\nAction: jump",
    "",
])
def test_parse_errors(text):
    p = parse_response(text, 1)
    assert p.parse_error is not None
    assert p.actions == []


def test_parse_bug_report_and_thought():
    p = parse_response("Thought: the plate\nvanished oddly\nAction: 4\n"
                       "Bug report: plate disappears after delivery", 1)
    assert p.thought == "the plate vanished oddly"
    assert p.bug_report == "plate disappears after delivery"
    assert parse_response("Thought: x\nAction: 1\nBug report: None", 1).bug_report is None
    assert parse_response("Thought: x\nAction: 1\nBug report: none observed", 1).bug_report is None


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_mock_react_run_parsing_fallback_and_reports(tmp_path):
    script = [
        "Thought: start\nAction: 0",                               # ok
        "blah blah no format",                                     # malformed -> wait
        "Thought: ok again\nAction: 1\nBug report: the floor ate my tomato",
        "Thought: x\nAction: 9",                                   # malformed -> wait
        "Thought: fine\nAction: 2",
    ]
    llm = MockLLM(script=script)
    agent = ReActAgent(llm, history=10)
    summary = run(agent, "V2_task_01_tomato", 0, Budgets(max_env_steps=5), tmp_path, CFG,
                  ignore_bug_ids=())
    steps = _read(tmp_path / "steps.jsonl")
    calls = _read(tmp_path / "llm_calls.jsonl")
    assert [s["action"] for s in steps] == [0, 4, 1, 4, 2]
    assert [c["parse_error"] is not None for c in calls] == [False, True, False, True, False]
    assert [s["llm_call_id"] for s in steps] == [0, 1, 2, 3, 4]
    # the turn right after a parse error carries the one-line format reminder, others do not
    reminder = FORMAT_REMINDER.split(":")[0]
    assert [reminder in u for _, u in llm.calls] == [False, False, True, False, True]
    assert summary["parse_error_rate"] == pytest.approx(0.4)
    assert summary["agent_bug_reports"] == 1
    reports = _read(tmp_path / "bug_reports.jsonl")
    assert reports[0]["report"] == "the floor ate my tomato"
    assert reports[0]["global_step"] == 2 and reports[0]["window"] == [-1, 5]
    assert "bug_ids_in_window" in reports[0]
    # history holds the previous thought/action/result
    assert "Thought: start | Action: 0 | Result: you at (2,1)" in llm.calls[1][1]
    for c in calls:
        assert c["backend"] == "mock" and c["model"] == "mock"


def test_multi_action_calls(tmp_path):
    llm = MockLLM(script=["Thought: go\nAction: 0, 0, 3", "Thought: back\nAction: 2,2"])
    agent = ReActAgent(llm, history=10, actions_per_call=3)
    run(agent, "V2_task_01_tomato", 0, Budgets(max_env_steps=5), tmp_path, CFG)
    steps = _read(tmp_path / "steps.jsonl")
    assert [s["action"] for s in steps] == [0, 0, 3, 2, 2]
    assert [s["llm_call_id"] for s in steps] == [0, 0, 0, 1, 1]
    assert len(llm.calls) == 2
    # the second prompt shows the observation after the LAST action of the first call
    assert "YOU: at (3,1), holding plate" in llm.calls[1][1]
    assert "Action: 0,0,3" in llm.calls[1][1]


def test_history_window_and_reset_marker(tmp_path):
    llm = MockLLM(script=["Thought: idle\nAction: 4"])
    agent = ReActAgent(llm, history=3)
    run(agent, "V2_task_01_tomato", 0, Budgets(max_env_steps=203), tmp_path, CFG)
    last = llm.calls[-1][1]
    assert "=== EPISODE RESET ===" in last
    assert "(episode 2)" in last
    assert last.count("[turn ") == 3
    assert "STEP: 2/200" in last
    assert "=== EPISODE RESET ===" not in llm.calls[150][1]


def test_context_guard_drops_history(tmp_path):
    warnings = []
    llm = MockLLM(script=["Thought: " + "word " * 200 + "\nAction: 4"], context_limit=1300)
    agent = ReActAgent(llm, history=10, warn=warnings.append)
    run(agent, "V2_task_01_tomato", 0, Budgets(max_env_steps=12), tmp_path, CFG)
    calls = _read(tmp_path / "llm_calls.jsonl")
    assert min(c["history_turns_used"] for c in calls) < 10
    assert warnings and "context guard" in warnings[0]
    from react_baseline.llm import estimate_tokens
    for system, user in llm.calls:
        assert estimate_tokens(system + user) <= 1300
        assert system == agent.system  # system prompt is never cut


def test_context_guard_errors_when_nothing_fits(tmp_path):
    llm = MockLLM(script=["Thought: x\nAction: 4"], context_limit=100)
    agent = ReActAgent(llm)
    with pytest.raises(ContextOverflowError):
        run(agent, "V2_task_01_tomato", 0, Budgets(max_env_steps=3), tmp_path, CFG)
    # logs are still finalised, with the error as the stop reason
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["stop_reason"].startswith("error: ContextOverflowError")


def test_llm_and_token_budgets(tmp_path):
    llm = MockLLM()
    s = run(ReActAgent(llm), "V2_task_01_tomato", 0, Budgets(max_env_steps=100, max_llm_calls=7),
            tmp_path / "a", CFG)
    assert s["llm_calls"] == 7 and s["total_env_steps"] == 7 and s["stop_reason"] == "max_llm_calls"
    s = run(ReActAgent(MockLLM()), "V2_task_01_tomato", 0, Budgets(max_env_steps=100, max_tokens=3000),
            tmp_path / "b", CFG)
    assert s["stop_reason"] == "max_tokens" and s["total_tokens"] >= 3000
    assert s["llm_calls"] < 100
