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
