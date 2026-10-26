"""Post-hoc rescoring with the normal-play sets: added fields only, summary.json untouched."""
import json

from react_baseline import rescore
from react_baseline.agents import ReActAgent
from react_baseline.llm import MockLLM
from react_baseline.repro import ReproConfig
from react_baseline.runner import Budgets, run

NORMAL = {"V2_task_07_tomato_steak": [1, 3, 4, 5, 6, 21, 68, 78, 127]}


def _fake_run(tmp_path):
    d = tmp_path / "react_x_V2_task_07_tomato_steak_seed0"
    d.mkdir()
    summary = {"agent": "react", "task": "V2_task_07_tomato_steak", "seed": 0, "n_unique_bugs": 5,
               "unique_bug_ids": [4, 20, 21, 78, 127], "deliveries_success": 2, "llm_calls": 10,
               "usd_cost": 0.01, "reports_strict_match": 3, "unique_bugs_reported_strict": 3,
               "unique_bugs_excluding_normal_play": 99, "repro_success_rate": 0.5}
    (d / "summary.json").write_text(json.dumps(summary))
    reports = [{"report": "a", "bug_ids_strict": [78]}, {"report": "b", "bug_ids_strict": [20, 21]},
               {"report": "c", "bug_ids_strict": []}, {"report": "d", "bug_ids_strict": [127, 4]}]
    (d / "bug_reports.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reports))
    attempts = [{"attempt_id": 0, "source_bug_ids_strict": [78], "reproduced_bug_ids": [78], "oracle_reproduced": True},
                {"attempt_id": 1, "source_bug_ids_strict": [20, 21], "reproduced_bug_ids": [21],
                 "oracle_reproduced": True},
                {"attempt_id": 2, "source_bug_ids_strict": [20], "reproduced_bug_ids": [20], "oracle_reproduced": True},
                {"attempt_id": 3, "source_bug_ids_strict": [], "reproduced_bug_ids": [], "oracle_reproduced": None}]
    (d / "repro_attempts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attempts))
    return d, summary


def test_adjusted_fields(tmp_path):
    d, summary = _fake_run(tmp_path)
    before = (d / "summary.json").read_bytes()
    out = rescore.rescore_run(d, NORMAL)
    assert (d / "summary.json").read_bytes() == before            # never overwritten
    saved = json.loads((d / "summary_rescored.json").read_text())
    assert saved == out
    assert all(saved[k] == v for k, v in summary.items() if k not in ("unique_bugs_excluding_normal_play",))
    assert out["unique_bugs_excluding_normal_play"] == 1           # only 20 is outside the set
    assert out["unique_bug_ids_excluding_normal_play"] == [20]
    assert out["reports_strict_match_adj"] == 1 and out["unique_bugs_reported_strict_adj"] == 1
    adj = {a["attempt_id"]: a for a in out["repro_attempts_adj"]}
    assert adj[0]["source_bug_ids_strict_adj"] == [] and adj[0]["oracle_reproduced_adj"] is None
    assert adj[1]["source_bug_ids_strict_adj"] == [20] and adj[1]["oracle_reproduced_adj"] is False
    assert adj[2]["oracle_reproduced_adj"] is True and adj[3]["oracle_reproduced_adj"] is None
    assert out["repro_success_rate_adj"] == 0.5 and out["repro_attempts_scored_adj"] == 2
    assert sum(out["bugs_by_proposed_class_adj"].values()) == 1
    assert out["rescore"]["normal_play_file"].endswith("normal_play_bugs.yaml")


def test_unknown_task_gives_none(tmp_path):
    d, _ = _fake_run(tmp_path)
    out = rescore.rescore_run(d, {})
    assert out["unique_bugs_excluding_normal_play"] is None and out["repro_success_rate_adj"] is None


def test_cli_on_a_grid_dir_with_real_runs(tmp_path, capsys):
    grid = tmp_path / "grid"
    script = ["Thought: t\nAction: interact(0,1)\nBug report: the steak looks odd",
              "Thought: u\nAction: interact(1,0)", "Thought: v\nAction: wait(3)"]
    for seed in (0, 1):
        run(ReActAgent(MockLLM(script=script), action_space="intent"), "V2_task_07_tomato_steak", seed,
            Budgets(max_env_steps=60), grid / f"react_mock_V2_task_07_tomato_steak_seed{seed}",
            {"backend": "mock", "model": "mock"}, repro=ReproConfig(enabled=True, steps=10, max_fraction=0.5))
    (grid / "react_mock_V2_task_07_tomato_steak_seed0_stale1").mkdir()
    before = {p: p.read_bytes() for p in grid.glob("*/summary.json")}
    assert rescore.main([str(grid)]) == 0
    out = capsys.readouterr().out
    assert "| react | 07_tomato_steak | 2 |" in out and "bugs adj" in out
    assert {p: p.read_bytes() for p in grid.glob("*/summary.json")} == before
    assert len(list(grid.glob("*/summary_rescored.json"))) == 2
