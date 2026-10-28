"""Strict vs lenient bug-report attribution, and taxonomy counts (reporting only)."""
import json

from react_baseline.agents import RandomAgent, ReActAgent
from react_baseline.llm import MockLLM
from react_baseline.runner import Budgets, attribute_reports, run
from react_baseline.taxonomy import TAXONOMY_CSV, load_taxonomy, taxonomy_counts


def _rows(spec):
    """spec: list of (decision_id, bug_ids) per global step 1..n."""
    return [{"global_step": i + 1, "decision_id": d, "bug_ids_triggered": b}
            for i, (d, b) in enumerate(spec)]


def test_strict_vs_lenient_on_synthetic_rows():
    # decision 0 = steps 1-3 (bug 20 at step 2); decision 1 = step 4 (no bug);
    # decision 2 = steps 5-6 (bug 42 at step 6); decision 3 = step 7
    rows = _rows([(0, []), (0, [20]), (0, []), (1, []), (2, []), (2, [42]), (3, [])])
    reports = [
        {"decision_id": 1, "global_step": 3, "report": "sink acts weird"},   # about decision 0
        {"decision_id": 2, "global_step": 4, "report": "nothing really"},    # about decision 1
        {"decision_id": 3, "global_step": 6, "report": "dirty plate"},       # about decision 2
        {"decision_id": 0, "global_step": 0, "report": "first call"},        # no previous decision
    ]
    out = {r["decision_id"]: r for r in attribute_reports(reports, rows)}
    assert out[1]["bug_ids_strict"] == [20] and out[1]["evidence_steps"] == [1, 2, 3]
    assert out[1]["bug_ids_in_window"] == [20, 42] and out[1]["window"] == [0, 6]
    # lenient matches bug 20 (step 2 is within ±3 of step 4) and 42; strict does not
    assert out[2]["bug_ids_strict"] == [] and out[2]["evidence_steps"] == [4]
    assert out[2]["bug_ids_in_window"] == [20, 42]
    assert out[3]["bug_ids_strict"] == [42] and out[3]["evidence_steps"] == [5, 6]
    assert out[0]["bug_ids_strict"] == [] and out[0]["evidence_steps"] == []
    assert out[0]["bug_ids_in_window"] == [20]


def test_summary_report_counts(tmp_path):
    # walk right into the sink from (3,3) with empty hands: bug 20 (USE_SINK_WITHOUT_PLATE)
    script = ["Thought: go\nAction: interact(4,3)",
              "Thought: odd\nAction: wait\nBug report: the sink reacted to empty hands",
              "Thought: x\nAction: wait\nBug report: nothing happened"]
    s = run(ReActAgent(MockLLM(script=script), action_space="intent"), "V2_task_01_tomato", 0,
            Budgets(max_env_steps=7, max_llm_calls=3), tmp_path, {"backend": "mock", "model": "mock"})
    reps = [json.loads(line) for line in (tmp_path / "bug_reports.jsonl").read_text().splitlines()]
    assert [r["decision_id"] for r in reps] == [1, 2]
    assert reps[0]["bug_ids_strict"] == [20] and reps[1]["bug_ids_strict"] == []
    assert reps[1]["bug_ids_in_window"] == [20]   # lenient still credits the second report
    assert s["reports_strict_match"] == 1 and s["reports_lenient_match"] == 2
    assert s["unique_bugs_reported_strict"] == 1 and s["unique_bug_ids_reported_strict"] == [20]
    for r in reps:
        assert {"window", "bug_ids_in_window", "llm_call_id", "global_step", "report"} <= set(r)


def test_taxonomy_counts_and_warnings():
    tax = load_taxonomy()
    assert len(tax) == 165 and all(r["status"] == "proposed" for r in tax.values())
    unreachable = next(b for b, r in tax.items() if r["reachability"] == "unreachable_on_shipped_maps")
    c = taxonomy_counts([20, 42, 20, 999, unreachable], tax)
    assert sum(c["bugs_by_proposed_class"].values()) == 4
    assert c["bugs_by_proposed_class"]["A_transition_anomaly"] == 1
    assert c["bugs_by_proposed_class"]["B_no_transition_anomaly"] == 1
    assert c["bugs_by_family"]["unknown"] == 1
    assert any(f"bug {unreachable} " in w and "unreachable_on_shipped_maps" in w for w in c["taxonomy_warnings"])
    assert any("bug 999 is not in the draft taxonomy" in w for w in c["taxonomy_warnings"])


def test_taxonomy_is_reporting_only(tmp_path, monkeypatch):
    """Summary metrics are identical whether or not the taxonomy file exists."""
    def go(d):
        return run(RandomAgent(0), "V2_task_01_tomato", 0, Budgets(max_env_steps=400), tmp_path / d,
                   {"backend": None, "model": None})
    a = go("a")
    monkeypatch.setattr("react_baseline.taxonomy.TAXONOMY_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr("react_baseline.taxonomy.load_taxonomy.__defaults__", (tmp_path / "missing.csv",))
    b = go("b")
    assert b["bugs_by_proposed_class"] is None and a["bugs_by_proposed_class"]
    tax_keys = {"bugs_by_proposed_class", "bugs_by_family", "taxonomy_warnings", "wall_time_s"}
    assert {k: v for k, v in a.items() if k not in tax_keys} == \
        {k: v for k, v in b.items() if k not in tax_keys}
    assert (tmp_path / "a" / "steps.jsonl").read_text() == (tmp_path / "b" / "steps.jsonl").read_text()
    assert a["ignore_bug_ids"] == [5]
    assert sum(a["bugs_by_proposed_class"].values()) == a["n_unique_bugs"]
    assert TAXONOMY_CSV.name == "bug_taxonomy_draft.csv"
