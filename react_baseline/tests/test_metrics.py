"""Loop metrics (reporting only)."""
import pytest

from react_baseline.agents import ReActAgent
from react_baseline.llm import MockLLM
from react_baseline.runner import Budgets, decision_keys, loop_metrics, run


@pytest.mark.parametrize("keys,loop,cyc", [
    ([], None, None),
    (list("abcdef"), 0.0, 0.0),
    (list("aaab"), 0.75, 0.0),
    (list("aab"), 0.0, 0.0),
    (list("abab"), 0.0, 1.0),
    (list("aba"), 0.0, 0.0),
    (list("xababababy"), 0.0, 0.8),
    (list("ababcccc"), 0.5, 0.5),
    (list("aaabab"), 0.5, 4 / 6),   # a,b,a,b at positions 2..5
])
def test_loop_metrics(keys, loop, cyc):
    m = loop_metrics(keys)
    if loop is None:
        assert m == {"loop_decision_fraction": None, "cycle2_decision_fraction": None}
    else:
        assert m["loop_decision_fraction"] == pytest.approx(loop)
        assert m["cycle2_decision_fraction"] == pytest.approx(cyc)


def test_decision_keys_group_intents():
    rows = [{"decision_id": 0, "intent_id": 0, "intent": "goto(1,3)", "action": 1},
            {"decision_id": 0, "intent_id": 0, "intent": "goto(1,3)", "action": 1},
            {"decision_id": 0, "intent_id": 1, "intent": "interact(0,3)", "action": 2},
            {"decision_id": 1, "intent_id": 2, "intent": "left", "action": 2}]
    assert decision_keys(rows) == ["goto(1,3); interact(0,3)", "left"]


def test_summary_has_loop_metrics(tmp_path):
    juggle = ["Thought: x\nAction: interact(0,2)"]     # the 14b pilot's tomato juggling
    s = run(ReActAgent(MockLLM(script=juggle), action_space="intent"), "V2_task_01_tomato", 0,
            Budgets(max_env_steps=20), tmp_path / "a", {"backend": "mock", "model": "mock"})
    assert s["loop_decision_fraction"] == 1.0 and s["cycle2_decision_fraction"] == 0.0
    alt = ["Thought: x\nAction: goto(2,2)", "Thought: y\nAction: interact(2,2)"]   # the 7b's loop
    s = run(ReActAgent(MockLLM(script=alt), action_space="intent"), "V2_task_01_tomato", 0,
            Budgets(max_env_steps=20), tmp_path / "b", {"backend": "mock", "model": "mock"})
    assert s["cycle2_decision_fraction"] == 1.0 and s["loop_decision_fraction"] == 0.0
