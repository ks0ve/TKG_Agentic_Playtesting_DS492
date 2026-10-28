"""Agent-driven reproduction: cycle, dedup, caps, no nesting, null scores, determinism, leaks."""
import json

import pytest

from react_baseline import pilot_check
from react_baseline.agents import RandomIntentAgent, ReActAgent
from react_baseline.llm import MockLLM
from react_baseline.repro import END_MARKER, START_MARKER, ReproConfig, near_duplicate, parse_reproduced
from react_baseline.runner import Budgets, run
from react_baseline.tests.test_prompt_leak import _leaks
from react_baseline.env_adapter import EnvAdapter  # noqa: F401  (sys.path for overcookedPlus)
from overcookedPlus.module.bug_manager import BugManager  # noqa: E402

SENTINEL = 987654   # emitted when bumping an empty counter with empty hands
CFG = {"backend": "mock", "model": "mock"}
REPORT = "the empty counter did something odd"
CYCLE = ["Thought: bump\nAction: interact(1,0)",                      # fires the sentinel
         f"Thought: odd\nAction: wait\nBug report: {REPORT}",         # report -> attempt
         "Thought: again\nAction: interact(1,0)",                     # (repro) fires again
         "Thought: done\nAction: wait\nReproduced: yes"]              # (repro) claim -> end


@pytest.fixture
def sentinel(monkeypatch):
    monkeypatch.setattr(BugManager, "check_pickup_from_empty", lambda self: [SENTINEL])


def _read(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def _go(tmp_path, script, env_steps=60, name="r", **cfg):
    llm = MockLLM(script=script)
    rc = ReproConfig(enabled=True, **{"steps": 10, "max_fraction": 1.0, **cfg})
    s = run(ReActAgent(llm, history=20, action_space="intent"), "V2_task_01_tomato", 0,
            Budgets(max_env_steps=env_steps), tmp_path / name, CFG, ignore_bug_ids=(), repro=rc)
    return s, llm, tmp_path / name


def test_parse_and_dedup_helpers():
    assert parse_reproduced("Thought: x\nAction: wait\nReproduced: yes")
    assert parse_reproduced("**Reproduced:** Yes")
    assert not parse_reproduced("Reproduced: no") and not parse_reproduced("I reproduced it")
    assert near_duplicate("The plate vanished!", "the plate  vanished", 0.8)
    assert near_duplicate("plate vanished after delivery today", "plate vanished after delivery", 0.8)
    assert not near_duplicate("plate vanished", "knife ate the tomato", 0.8)


def test_full_attempt_cycle(tmp_path, sentinel):
    s, llm, d = _go(tmp_path, CYCLE, env_steps=30)
    att = _read(d / "repro_attempts.jsonl")
    assert len(att) == 1
    a = att[0]
    assert a["source_report"] == REPORT and a["source_decision_id"] == 1
    assert a["source_bug_ids_strict"] == [SENTINEL]
    assert a["oracle_reproduced"] is True and a["agent_claimed_reproduced"] is True
    assert a["end_reason"] == "claimed" and a["reproduced_bug_ids"] == [SENTINEL]
    assert a["first_repro_step"] is not None and a["steps"] >= 2 and a["llm_calls"] == 2
    steps = _read(d / "steps.jsonl")
    rep = [r for r in steps if r["phase"] == "repro"]
    assert rep and all(r["repro_attempt_id"] == 0 for r in rep)
    assert a["steps"] == len(rep)
    eps = _read(d / "episodes.jsonl")
    assert [e["end_reason"] for e in eps][:2] == ["repro_start", "repro_end"]
    assert eps[1]["phase"] == "repro"
    calls = _read(d / "llm_calls.jsonl")
    assert [c["phase"] for c in calls[:5]] == ["explore", "explore", "repro", "repro", "explore"]
    assert calls[3]["parsed_reproduced"] is True
    # what the agent saw: reset, marker, task block (with steps left), then the end marker
    u2 = llm.calls[2][1]
    assert START_MARKER in u2 and "YOU: at (1,1)" in u2
    assert f'REPRODUCTION TASK: reproduce the defect you reported: "{REPORT}". Steps left: 10.' in u2
    assert "Steps left: " in llm.calls[3][1] and "Steps left: 10." not in llm.calls[3][1]
    u4 = llm.calls[4][1]
    assert END_MARKER in u4 and "REPRODUCTION TASK" not in u4
    assert s["repro_attempts"] == 1 and s["repro_success_rate"] == 1.0
    assert s["agent_claim_accuracy"] == 1.0 and s["unique_bugs_reproduced"] == 1
    assert s["steps_repro"] == len(rep) and s["steps_explore"] == 30 - len(rep)
    assert s["llm_calls_repro"] == 2 and s["llm_calls_explore"] == s["llm_calls"] - 2
    # the cycle repeats: the same report again is a duplicate, not a second attempt
    assert s["repro_skipped"].get("duplicate", 0) >= 1
    ok, lines = pilot_check.check(d)
    assert any("replay mismatches vs steps.jsonl: 0" in line for line in lines), lines


def test_unmatched_report_scores_null(tmp_path, sentinel):
    script = [f"Thought: hm\nAction: wait\nBug report: {REPORT}", "Thought: x\nAction: wait"]
    s, _, d = _go(tmp_path, script, env_steps=20)
    a = _read(d / "repro_attempts.jsonl")[0]
    assert a["source_bug_ids_strict"] == [] and a["oracle_reproduced"] is None
    assert a["end_reason"] == "steps" and a["steps"] == 10
    assert s["repro_success_rate"] is None and s["agent_claim_accuracy"] is None


def test_caps_attempts_and_fraction(tmp_path, sentinel):
    script = ["Thought: a\nAction: wait\nBug report: the sink is strange",
              "Thought: b\nAction: wait\nBug report: knife refuses onions completely",
              "Thought: c\nAction: wait\nBug report: plate teleports to window"]
    s, _, _ = _go(tmp_path, script, env_steps=60, name="att", max_attempts=1, max_fraction=1.0, steps=5)
    assert s["repro_attempts"] == 1 and s["repro_skipped"].get("max_attempts", 0) >= 1
    s, _, _ = _go(tmp_path, script, env_steps=60, name="frac", max_fraction=0.1, steps=10)
    assert s["repro_attempts"] == 0 and s["repro_skipped"]["max_fraction"] >= 1


def test_no_nested_attempts(tmp_path, sentinel):
    script = [f"Thought: a\nAction: wait\nBug report: {REPORT}",
              "Thought: b\nAction: wait\nBug report: a completely different defect with the sink"]
    s, _, d = _go(tmp_path, script, env_steps=30)
    att = _read(d / "repro_attempts.jsonl")
    reps = _read(d / "bug_reports.jsonl")
    in_repro = [r for r in reps if r["phase"] == "repro"]
    assert in_repro                                    # reports during an attempt are logged ...
    assert all(a["source_decision_id"] not in {r["decision_id"] for r in in_repro} for a in att)
    assert len(att) == 2                                # ... but only explore reports start attempts
    assert all(a["source_report"] in (REPORT, script[1].split("Bug report: ")[1]) for a in att)


def test_repro_deterministic_with_mock(tmp_path, sentinel):
    _, _, a = _go(tmp_path, CYCLE, env_steps=60, name="a")
    _, _, b = _go(tmp_path, CYCLE, env_steps=60, name="b")
    for f in ("steps.jsonl", "repro_attempts.jsonl", "episodes.jsonl", "transitions.jsonl"):
        x, y = (a / f).read_text(), (b / f).read_text()
        if f == "repro_attempts.jsonl":
            x, y = (json.dumps([{k: v for k, v in r.items()} for r in map(json.loads, t.splitlines())])
                    for t in (x, y))
        assert x == y, f


def test_random_agents_never_reproduce(tmp_path):
    s = run(RandomIntentAgent(0), "V2_task_01_tomato", 0, Budgets(max_env_steps=100), tmp_path,
            {"backend": None, "model": None}, repro=ReproConfig(enabled=True))
    assert s["repro_attempts"] == 0 and s["steps_repro"] == 0 and s["steps_explore"] == 100
    assert (tmp_path / "repro_attempts.jsonl").read_text() == ""


def test_repro_prompt_block_has_no_oracle(tmp_path, monkeypatch):
    from react_baseline.tests.test_prompt_leak import _install_sentinels
    _install_sentinels(monkeypatch)
    script = ["Thought: bump\nAction: interact(1,0)",
              "Thought: odd\nAction: wait\nBug report: picking from an empty counter misbehaves",
              "Thought: again\nAction: interact(1,0)", "Thought: plate\nAction: interact(3,0)",
              "Thought: deliver\nAction: interact(2,2)", "Thought: done\nAction: wait\nReproduced: yes"]
    llm = MockLLM(script=script)
    run(ReActAgent(llm, history=50, action_space="intent"), "V2_task_01_tomato", 0,
        Budgets(max_env_steps=40), tmp_path, CFG, ignore_bug_ids=(),
        repro=ReproConfig(enabled=True, steps=20, max_fraction=1.0))
    fired = {b for r in (json.loads(x) for x in (tmp_path / "steps.jsonl").read_text().splitlines())
             for b in r["bug_ids_triggered"]}
    assert SENTINEL in fired   # the oracle fired while the agent was prompted to reproduce
    repro_prompts = [u for _, u in llm.calls if "REPRODUCTION TASK" in u]
    assert repro_prompts
    for _, u in llm.calls:
        assert _leaks(u) == [], u
