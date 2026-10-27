"""Non-reports on the 'Bug report:' line are discarded (logged, never counted), and the system
prompt says when to use the line. Texts are the exact reports 1-3 of the v2.1 salad Haiku pilot
(runs/pilot/20261026-201139_react_V2_task_11_tomato_lettuce_onion_seed0/bug_reports.jsonl)."""
import json

import pytest

from react_baseline import prompts
from react_baseline.agents import ReActAgent, parse_intent_response, parse_response
from react_baseline.llm import MockLLM
from react_baseline.repro import ReproConfig
from react_baseline.runner import Budgets, run

# report 1: a real (suspected) defect, must still count
REAL = 'After a rejected delivery of an incomplete plate (chopped onion only), the held plate and its contents vanished from my hands: the observation shows me holding nothing and the plate back on the counter at (4,3) as dirty and emp'
# report 2: says an earlier report is uncertain
UNCONFIRMED = 'Unconfirmed. My episode 1 report of a plate vanishing after a rejected delivery may be a misread of the observation; it needs reproduction before it is treated as a defect.'
# report 3: says no defect is confirmed
NO_DEFECT = 'The history at turn 33 says the plate with chopped onion was observed on the counter at (4,3) and at turn 39 the tomato was placed on it, but the earlier turn-32 note placed the plate at (4,3) while the current map still lists an empty counter sitting at (4,2)\'s neighbour positions inconsistently; the cutting board at (4,2) is shown as "K" in the map while the adjacency list calls (2,2) a delivery window, which is consistent, so no clear defect is confirmed here.'

SENTENCE = ("Only add a Bug report line when you believe you observed a defect. Never use it to say "
            "that no defect was found or that an earlier report is uncertain.")


def test_system_prompt_format_sentence():
    for space in ("primitive", "intent"):
        s = prompts.system_prompt(1, space)
        assert SENTENCE + "\nBug report: <one-sentence description of the suspected bug>" in s
        assert "Optionally add one more line" not in s


@pytest.mark.parametrize("parse", [parse_response, parse_intent_response])
def test_pilot_reports_2_and_3_are_discarded_report_1_counts(parse):
    for text in (UNCONFIRMED, NO_DEFECT):
        p = parse(f"Thought: t\nAction: wait\nBug report: {text}")
        assert p.bug_report is None and p.bug_report_discarded == text
    p = parse(f"Thought: t\nAction: wait\nBug report: {REAL}")
    assert p.bug_report == REAL and p.bug_report_discarded is None


@pytest.mark.parametrize("text,discarded", [
    ("No defect observed this turn.", True), ("no bug here", True), ("Nothing unusual.", True),
    ("Not a bug: the plate was dirty after delivery.", True), ("**Unconfirmed** plate issue", True),
    ("The steak burned, but no bug is found in the timing.", True),
    ("No clear bug observed.", True), ("None", False),          # "None" stays a plain absence
    ("The plate vanished after delivery; no defect warning was shown.", False),
    ("Delivered plate was rejected although it matched the order.", False),
])
def test_non_report_patterns(text, discarded):
    p = parse_response(f"Thought: t\nAction: 0\nBug report: {text}")
    assert (p.bug_report_discarded is not None) is discarded   # (leading markdown is stripped)
    assert p.bug_report is (None if discarded or text == "None" else p.bug_report)


def test_run_logs_discarded_reports_and_counts_them(tmp_path):
    script = [f"Thought: a\nAction: wait\nBug report: {UNCONFIRMED}",
              f"Thought: b\nAction: wait\nBug report: {NO_DEFECT}",
              f"Thought: c\nAction: wait\nBug report: {REAL}",
              "Thought: d\nAction: wait"]
    s = run(ReActAgent(MockLLM(script=script), action_space="intent"), "V2_task_11_tomato_lettuce_onion", 0,
            Budgets(max_env_steps=4), tmp_path, {"backend": "mock", "model": "mock"},
            repro=ReproConfig(enabled=True, steps=1, max_fraction=0.5))
    calls = [json.loads(l) for l in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    assert [c["bug_report_discarded"] for c in calls[:3]] == [UNCONFIRMED, NO_DEFECT, None]
    assert [c["parsed_bug_report"] for c in calls[:3]] == [None, None, REAL]
    reports = [json.loads(l) for l in (tmp_path / "bug_reports.jsonl").read_text().splitlines()]
    assert [r["report"] for r in reports] == [REAL]
    assert s["agent_bug_reports"] == 1 and s["bug_reports_discarded"] == 2
    attempts = [json.loads(l) for l in (tmp_path / "repro_attempts.jsonl").read_text().splitlines()]
    assert [a["source_report"] for a in attempts] == [REAL]   # non-reports never start reproduction
