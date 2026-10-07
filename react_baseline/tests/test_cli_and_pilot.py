import json

from react_baseline import pilot_check, run as run_cli


def _dirs(out):
    return sorted(p for p in out.iterdir() if p.is_dir())


def test_cli_multiple_seeds(tmp_path, capsys):
    rc = run_cli.main(["--agent", "random", "--seeds", "0,1", "--max-env-steps", "60",
                       "--out-dir", str(tmp_path), "--quiet"])
    assert rc == 0
    dirs = _dirs(tmp_path)
    assert len(dirs) == 2
    seeds = sorted(json.loads((d / "config.json").read_text())["seed"] for d in dirs)
    assert seeds == [0, 1]
    assert "AGGREGATE over seeds [0, 1]" in capsys.readouterr().out


def test_cli_default_temperature(tmp_path):
    args = run_cli.parse_args(["--agent", "react"])
    assert args.temperature == 0.7 and args.llm == "ollama"


def test_pilot_check_passes_random_and_fails_stuck_agent(tmp_path):
    assert run_cli.main(["--agent", "random", "--max-env-steps", "200", "--out-dir",
                         str(tmp_path / "ok"), "--quiet"]) == 0
    ok, lines = pilot_check.check(_dirs(tmp_path / "ok")[0])
    assert ok, lines

    # an agent that only ever presses "down" (run #1's failure mode)
    from react_baseline.agents import ReActAgent
    from react_baseline.llm import MockLLM
    from react_baseline.runner import Budgets, run
    d = tmp_path / "stuck"
    run(ReActAgent(MockLLM(script=["Thought: get tomato\nAction: 1"])), "V2_task_01_tomato", 0,
        Budgets(max_env_steps=50), d, {"backend": "mock", "model": "mock"})
    ok, lines = pilot_check.check(d)
    assert not ok
    assert any(line.startswith("[FAIL] longest same-action streak: 50") for line in lines)
