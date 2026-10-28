"""Reference rules from the game manual: coverage of every interaction family, unknowns, provenance."""
from pathlib import Path

import pytest

from memory.reference import RULES_YAML, Reference, coverage
from react_baseline.cheap_policy import CheapAgent
from react_baseline.runner import Budgets, run

REF = Reference()

CASES = [  # (ctx0, observed outcome, rule id, verdict)
    ("move | hands:chopped tomato | floor", "move | - | hands:chopped tomato | floor", "move_floor", "match"),
    ("wait | hands:nothing | none", "none | - | hands:nothing | none", "wait", "match"),
    ("move | hands:nothing | wall", "none | - | hands:nothing | wall", "bump_wall", "match"),
    ("interact | hands:nothing | counter:holding:unchopped tomato",
     "pickup | - | hands:unchopped tomato | counter:empty", "pickup_from_counter", "match"),
    ("interact | hands:unchopped tomato | counter:empty",
     "putdown | True | hands:nothing | counter:holding:unchopped tomato", "put_on_counter", "match"),
    ("interact | hands:unchopped onion | cutting board:empty",
     "putdown | True | hands:nothing | cutting board:holding:unchopped onion", "put_on_cutting_board", "match"),
    ("interact | hands:nothing | cutting board:holding:unchopped onion",
     "chop | True | hands:nothing | cutting board:holding:chopped onion", "chop", "match"),
    ("interact | hands:nothing | cutting board:holding:chopped onion",
     "pickup | - | hands:chopped onion | cutting board:empty", "take_from_board", "match"),
    ("interact | hands:nothing | sink:holding:plate:dirty:[]",
     "wash | - | hands:nothing | sink:holding:plate:clean:[]", "wash", "match"),
    ("interact | hands:plate:clean:[] | cutting board:holding:chopped tomato",
     "pickup_into_plate | True | hands:plate:clean:[chopped tomato] | cutting board:empty",
     "plate_add_from_station", "match"),
    ("interact | hands:plate:clean:[chopped tomato] | delivery window:empty",
     "deliver | True | hands:nothing | delivery window:empty", "deliver_plate", "match"),
    ("interact | hands:unchopped tomato | trash can:empty",
     "dump | - | hands:nothing | trash can:empty", "trash", "match"),
    ("interact | hands:chopped uncooked steak | pan:empty",
     "putdown | True | hands:nothing | pan:holding:chopped cooked steak", "put_steak_in_pan", "match"),
    # the game differs from the manual (a logged manual_mismatch)
    ("interact | hands:nothing | counter:holding:unchopped onion",
     "none | - | hands:nothing | counter:holding:unchopped onion", "pickup_from_counter", "mismatch"),
    # unknown: the manual says nothing
    ("interact | hands:nothing | counter:empty", "none | - | hands:nothing | counter:empty", None, "none"),
    ("interact | hands:plate:clean:[] | counter:holding:unchopped tomato",
     "pickup_into_plate | False | hands:plate:clean:[] | counter:holding:unchopped tomato", None, "none"),
]


@pytest.mark.parametrize("ctx0,outcome,rule,verdict", CASES)
def test_rules(ctx0, outcome, rule, verdict):
    assert REF.verdict(ctx0, outcome) == (rule, verdict)


def test_rule_families_and_flags():
    events = {str(r.expect["event"]) for r in REF.rules}
    assert {"move", "none", "pickup", "putdown", "chop", "wash", "pickup_into_plate", "deliver", "dump"} <= events
    assert {r.id for r in REF.rules if not r.changes_state} == {"wait", "bump_wall"}
    assert all(r.manual and r.says for r in REF.rules)
    assert REF.lookup("interact | hands:nothing | oven:empty") is None


def test_provenance_manual_only():
    header = RULES_YAML.read_text(encoding="utf-8").split(chr(10) + "rules:")[0]
    assert "ONLY from the GAME RULES text" in header and "were NOT consulted" in header
    src = (Path(__file__).resolve().parents[2] / "memory" / "reference.py").read_text(encoding="utf-8")
    for forbidden in ("bug_manager", "taxonomy", "normal_play"):
        assert forbidden not in src.replace("bug_manager.py, the taxonomy and normal_play_bugs were not consulted", "")
    assert len(REF.manual_mismatches) == 3


def test_coverage_report_on_stored_runs(tmp_path):
    for seed in (0, 1):
        run(CheapAgent(seed), "V2_task_07_tomato_steak", seed, Budgets(max_env_steps=300), tmp_path / f"r{seed}",
            {"backend": None, "model": None})
    rep = coverage(tmp_path)
    assert rep["transitions"] > 100 and 0.3 < rep["coverage"] < 1.0 and rep["match_rate"] > 0.9
    assert sum(v for k, v in rep["per_rule"].items()) == rep["covered"]
