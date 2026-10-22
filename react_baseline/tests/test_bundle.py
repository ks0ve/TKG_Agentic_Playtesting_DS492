"""Bundle tool end to end on mock runs."""
import zipfile

from react_baseline import bundle
from react_baseline.agents import RandomIntentAgent, ReActAgent
from react_baseline.llm import MockLLM
from react_baseline.repro import ReproConfig
from react_baseline.runner import Budgets, run


def test_bundle_mock_runs(tmp_path):
    script = ["Thought: bump\nAction: interact(1,0)",
              "Thought: odd\nAction: wait\nBug report: the empty counter did something odd",
              "Thought: again\nAction: interact(1,0)", "Thought: done\nAction: wait\nReproduced: yes"]
    a = tmp_path / "runs" / "react_mock"
    run(ReActAgent(MockLLM(script=script), action_space="intent"), "V2_task_01_tomato", 0,
        Budgets(max_env_steps=40), a, {"backend": "mock", "model": "mock"},
        repro=ReproConfig(enabled=True, steps=10, max_fraction=1.0))
    b = tmp_path / "runs" / "ri"
    run(RandomIntentAgent(0), "V2_task_01_tomato", 0, Budgets(max_env_steps=40), b,
        {"backend": None, "model": None})
    out = bundle.build([a, b, tmp_path / "nope"], tmp_path / "out" / "b.zip")
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        for d in ("react_mock", "ri"):
            for f in ("config.json", "summary.json", "steps.jsonl", "llm_calls.jsonl",
                      "bug_reports.jsonl", "repro_attempts.jsonl", "transitions.jsonl", "pilot_check.txt"):
                assert f"runs/{d}/{f}" in names, (d, f)
        assert {"spend_report.txt", "git.txt", "grid_dry_run.txt", "MANIFEST.txt"} <= names
        assert "frozen check" in z.read("grid_dry_run.txt").decode()
        assert "Project cap $20.00" in z.read("spend_report.txt").decode()
        assert "MISSING run dir" in z.read("MANIFEST.txt").decode()
        assert all("\\" not in n for n in names)


def test_latest_pilots_and_grid_runs(tmp_path):
    import json
    import yaml
    from react_baseline import frozen, grid

    pilots = tmp_path / "pilot"
    for name, agent, steps in [("20261001-000000_old", "random-intent", 20),
                               ("20261002-000000_new", "random-intent", 30),
                               ("20261001-120000_other", "random", 20)]:
        Agent = RandomIntentAgent if agent == "random-intent" else __import__(
            "react_baseline.agents", fromlist=["RandomAgent"]).RandomAgent
        run(Agent(0), "V2_task_01_tomato", 0, Budgets(max_env_steps=steps), pilots / name,
            {"backend": None, "model": None})
    picked = sorted(d.name for d in bundle.latest_pilot_dirs(pilots))
    assert picked == ["20261001-120000_other", "20261002-000000_new"]

    fz = tmp_path / "frozen.yaml"
    fz.write_text(yaml.safe_dump(frozen.code_identity()), encoding="utf-8")
    g = {"name": "mini", "frozen": str(fz), "out_dir": str(tmp_path / "grid"),
         "tasks": ["V2_task_01_tomato"], "max_env_steps": 20, "max_usd_per_paid_run": 0.5,
         "rows": [{"agent": "random", "seeds": [0, 1]}]}
    gy = tmp_path / "g.yaml"
    gy.write_text(yaml.safe_dump(g), encoding="utf-8")
    assert grid.main([str(gy), "--only", "random"]) == 0
    s = tmp_path / "grid" / "random_none_V2_task_01_tomato_seed1" / "summary.json"
    s.write_text(json.dumps({**json.loads(s.read_text()), "stop_reason": "interrupted"}))  # unfinished
    out = bundle.build([], tmp_path / "x.zip", gy, latest_pilots=True, grid_runs=True, pilot_root=pilots)
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
    assert "grid/random_none_V2_task_01_tomato_seed0/summary.json" in names
    assert not any("seed1" in n for n in names)                       # only finished grid runs
    assert "pilots/20261002-000000_new/steps.jsonl" in names
    assert not any("20261001-000000_old" in n for n in names)
    import pytest
    with pytest.raises(SystemExit):        # nothing selected: refuse rather than guess paths
        bundle.main(["-o", str(tmp_path / "y.zip")])
