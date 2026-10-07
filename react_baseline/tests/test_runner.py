import json

from react_baseline.agents import RandomAgent
from react_baseline.runner import Budgets, run

STEP_KEYS = {"global_step", "episode", "episode_step", "action", "reward", "done",
             "bug_ids_triggered", "new_unique_bug_ids", "bug_ids_raw", "delivery_success",
             "llm_call_id"}
SUMMARY_KEYS = {"total_env_steps", "episodes", "llm_calls", "input_tokens", "output_tokens",
                "total_tokens", "wall_time_s", "unique_bug_ids", "first_found_step",
                "discovery_curve", "deliveries_success", "agent_bug_reports", "parse_error_rate"}
CONFIG_KEYS = {"cli_args", "backend", "model", "git_commit", "sage_commit", "seed", "task",
               "ignore_bug_ids"}


def _random_run(tmp_path, name, seed=0, steps=500):
    d = tmp_path / name
    cfg = {"cli_args": {"agent": "random", "seed": seed}, "backend": None, "model": None}
    summary = run(RandomAgent(seed), "V2_task_01_tomato", seed, Budgets(max_env_steps=steps), d, cfg)
    return d, summary


def test_random_run_writes_documented_schema(tmp_path):
    d, summary = _random_run(tmp_path, "r0")
    for f in ["config.json", "steps.jsonl", "llm_calls.jsonl", "bug_reports.jsonl", "summary.json"]:
        assert (d / f).exists(), f
    rows = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == 500
    for r in rows:
        assert set(r) == STEP_KEYS
        assert r["llm_call_id"] is None
        assert 5 not in r["bug_ids_triggered"]  # default ignore list
        assert 0 <= r["action"] <= 4
    assert [r["global_step"] for r in rows] == list(range(1, 501))
    assert rows[0]["episode"] == 1 and rows[0]["episode_step"] == 1
    # episodes auto-reset after 200 steps
    assert rows[199]["done"] and rows[200]["episode"] == 2 and rows[200]["episode_step"] == 1
    assert (d / "llm_calls.jsonl").read_text() == ""
    summary_disk = json.loads((d / "summary.json").read_text())
    assert summary_disk == summary
    assert SUMMARY_KEYS <= set(summary)
    assert summary["total_env_steps"] == 500 and summary["episodes"] == 3
    assert summary["parse_error_rate"] is None
    # first_found_step and new_unique_bug_ids agree
    firsts = {str(b): r["global_step"] for r in rows for b in r["new_unique_bug_ids"]}
    assert firsts == summary["first_found_step"]
    assert summary["discovery_curve"][-1]["unique_bugs"] == len(summary["unique_bug_ids"])
    cfg = json.loads((d / "config.json").read_text())
    assert CONFIG_KEYS <= set(cfg)


def test_random_run_is_deterministic(tmp_path):
    a, _ = _random_run(tmp_path, "a", seed=0)
    b, _ = _random_run(tmp_path, "b", seed=0)
    c, _ = _random_run(tmp_path, "c", seed=1)
    sa = (a / "steps.jsonl").read_text()
    assert sa == (b / "steps.jsonl").read_text()
    assert sa != (c / "steps.jsonl").read_text()


def test_random_agent_independent_of_global_random(tmp_path):
    """The env reseeds the global RNG to 42 on every reset; episodes must still differ."""
    d, _ = _random_run(tmp_path, "e", steps=400)
    rows = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
    ep1 = [r["action"] for r in rows if r["episode"] == 1]
    ep2 = [r["action"] for r in rows if r["episode"] == 2]
    assert ep1 != ep2
