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


def test_tkg_lookup_order_observed_then_reference_then_unseen():
    import json as _json
    from memory.tkg import TKG
    tkg = TKG(min_support=3, reference=REF)
    c0 = "interact | hands:nothing | counter:holding:unchopped tomato"
    got = "pickup | - | hands:unchopped tomato | counter:empty"
    first = tkg.lookup("START || " + c0, c0)
    assert first["level"] == "reference" and first["id"] == "exp:reference:pickup_from_counter"
    unknown = "interact | hands:nothing | counter:empty"
    assert tkg.lookup("START || " + unknown, unknown)["level"] == "unseen"
    for i in range(3):
        t = {"id": f"t{i}", "episode_key": f"e{i}", "global_step_start": i, "global_step_end": i, "ctx0": c0,
             "ctx1": "START || " + c0, "outcome": got, "target": "counter:holding:unchopped tomato"}
        tkg.judge_reference(t)
        assert (t["reference_rule_id"], t["reference_verdict"]) == ("pickup_from_counter", "match")
        tkg.add(t, None, None, first["id"])
    after = tkg.lookup("START || " + c0, c0)
    assert after["level"] == "ctx1" and after["support"] == 3          # observed wins once supported
    node = tkg.expectations["exp:reference:pickup_from_counter"].to_node()
    assert node["source"] == "reference" and node["outcomes"] == {"match": 3} and node["changes_state"] is True
    edges = [e for e in tkg.export()["edges"] if e["type"] == "ASSESSED_AGAINST"]
    assert {e["target"] for e in edges} == {"exp:reference:pickup_from_counter"}
    assert _json.dumps(tkg.export())                                   # serialisable


def test_live_reference_fields_and_summary(tmp_path):
    import json as _json
    from react_baseline.llm import MockIntentLLM
    from react_baseline.tkg_agent import TKGAgent
    s = run(TKGAgent(MockIntentLLM(seed=1), seed=1), "V2_task_07_tomato_steak", 1, Budgets(max_env_steps=400),
            tmp_path, {"backend": "mock", "model": "mock"})
    rows = [_json.loads(l) for l in (tmp_path / "transitions.jsonl").read_text().splitlines()]
    assert {r["reference_verdict"] for r in rows} <= {"match", "mismatch", "none"}
    covered = [r for r in rows if r["reference_verdict"] != "none"]
    assert covered and all(r["reference_rule_id"] for r in covered)
    assert s["reference_coverage"] == len(covered) / len(rows)
    assert s["reference_match_rate"] == sum(r["reference_verdict"] == "match" for r in covered) / len(covered)
    assert s["reference_mismatches"] == sum(m["count"] for m in s["manual_mismatches"]) or len(s["manual_mismatches"]) == 20
    graph = _json.loads((tmp_path / "tkg.json").read_text())
    ref_nodes = {n["id"] for n in graph["nodes"] if n["type"] == "Expectation" and n["source"] == "reference"}
    assert ref_nodes and any(e["type"] == "ASSESSED_AGAINST" and e["target"] in ref_nodes for e in graph["edges"])
