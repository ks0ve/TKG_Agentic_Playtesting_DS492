"""Grid driver (resumable, dry-run, guarded) and the frozen baseline config."""
import json
from pathlib import Path

import pytest
import yaml

from react_baseline import frozen, grid, llm

REPO = Path(__file__).resolve().parents[2]


def test_shipped_frozen_config_matches_code():
    assert frozen.check(REPO / "configs" / "frozen" / "baseline_v2_1.yaml") == []
    assert frozen.FROZEN_DEFAULT.name == "baseline_v2_1.yaml"


def test_v2_0_frozen_config_is_superseded_by_wording_only():
    """baseline_v2.yaml stays as the record of the v2.0 runs; v2.1 changed only agent-visible text."""
    bad = frozen.check(REPO / "configs" / "frozen" / "baseline_v2.yaml")
    assert [b.split(":")[0] for b in bad] == ["prompts_sha256", "renderer_sha256"]
    v2, v21 = (frozen.load(REPO / "configs" / "frozen" / f) for f in ("baseline_v2.yaml", "baseline_v2_1.yaml"))
    for key in ("action_reader_version", "action_reader", "agent", "repro", "budget", "tasks", "seeds",
                "ignore_bug_ids"):
        assert v2[key] == v21[key], key


def test_frozen_mismatch_detected(tmp_path):
    f = tmp_path / "frozen.yaml"
    f.write_text("prompts_sha256: deadbeef\naction_reader_version: 0.1\n", encoding="utf-8")
    bad = frozen.check(f)
    assert len(bad) == 3 and "prompts_sha256" in bad[0] and "renderer_sha256" in bad[1]


def test_renderer_hash_tracks_event_wording(monkeypatch):
    from react_baseline import env_adapter
    h = frozen.renderer_hash()
    monkeypatch.setattr(env_adapter, "display_name", lambda raw: raw)   # the old internal names
    assert frozen.renderer_hash() != h


def test_shipped_grid_combinations():
    g = grid.load_grid(REPO / "configs" / "grids" / "baseline_v2.yaml")
    cs = grid.combos(g)
    assert len(cs) == 15 + 9 + 3 + 9 + 9
    haiku = [c for c in cs if c.profile == "haiku-5.5" and c.agent == "react"]
    assert len(haiku) == 15 and all(c.paid for c in haiku)
    a = haiku[0].argv
    assert "--allow-paid" in a and a[a.index("--max-usd") + 1] == "1.00"
    assert "--temperature" not in a and "--repro" in a and a[a.index("--max-env-steps") + 1] == "1000"
    assert a[a.index("--repro-steps") + 1] == "50"
    llama = next(c for c in cs if c.profile == "llama3.1-8b" and c.agent == "react")
    assert llama.argv[llama.argv.index("--temperature") + 1] == "0.7" and not llama.paid
    raw = [c for c in cs if c.agent == "react-raw"]
    assert len(raw) == 3 and {c.profile for c in raw} == {"haiku-5.5"}
    assert {c.task for c in raw} == {"V2_task_01_tomato"} and all(c.paid for c in raw)
    assert not any(c.agent == "react-raw" and c.profile == "llama3.1-8b" for c in cs)
    assert len({c.config_hash for c in cs}) == len(cs)
    assert {c.task for c in cs} == {"V2_task_01_tomato", "V2_task_07_tomato_steak",
                                    "V2_task_11_tomato_lettuce_onion"}


def test_progress_grid_is_a_subset_with_identical_hashes():
    main = {c.label: c for c in grid.combos(grid.load_grid(REPO / "configs/grids/baseline_v2.yaml"))}
    prog = grid.combos(grid.load_grid(REPO / "configs/grids/baseline_v2_progress.yaml"))
    assert len(prog) == 9 + 3 + 9 + 9 + 3
    for c in prog:
        m = main[c.label]   # every progress run exists in the main grid ...
        assert (c.run_dir, c.config_hash, c.argv) == (m.run_dir, m.config_hash, m.argv)  # ... identically
    assert {c.task for c in prog if c.profile == "llama3.1-8b"} == {"V2_task_01_tomato"}


def test_only_accepts_comma_list():
    g = grid.load_grid(REPO / "configs/grids/baseline_v2_progress.yaml")
    cs = grid.combos(g, "react/haiku-5.5,react-raw/haiku-5.5")
    assert {(c.agent, c.profile) for c in cs} == {("react", "haiku-5.5"), ("react-raw", "haiku-5.5")}
    cs = grid.combos(g, "react/llama3.1-8b,random,random-intent")
    assert not any(c.paid for c in cs) and len(cs) == 3 + 9 + 9


@pytest.fixture
def mini_grid(tmp_path, monkeypatch):
    y = tmp_path / "models.yaml"
    y.write_text("defaults: {mock: mock, ollama: mock}\nprofiles:\n  mock: {backend: mock, model: mock}\n"
                 "  mock-paid: {backend: mock, model: mock-paid, price_in_per_mtok: 1.0, "
                 "price_out_per_mtok: 5.0}\n", encoding="utf-8")
    monkeypatch.setattr(llm, "MODELS_YAML", y)
    fz = tmp_path / "frozen.yaml"
    fz.write_text(yaml.safe_dump(frozen.code_identity()), encoding="utf-8")
    g = {"name": "mini", "frozen": str(fz), "out_dir": str(tmp_path / "grid"),
         "tasks": ["V2_task_01_tomato"], "max_env_steps": 30, "history": 10, "temperature": 0.7,
         "intents_per_call": 1, "max_macro_len": 40, "max_usd_per_paid_run": 0.5,
         "repro": {"enabled": True, "steps": 10, "max_attempts": 2, "max_fraction": 0.5,
                   "dedup_jaccard": 0.8},
         "rows": [{"agent": "random-intent", "seeds": [0, 1]},
                  {"agent": "react", "profile": "mock", "seeds": [0]},
                  {"agent": "react", "profile": "mock-paid", "seeds": [0]}]}
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(g), encoding="utf-8")
    return path, tmp_path, fz


def test_grid_runs_resumes_and_reruns_stale(mini_grid, capsys):
    path, tmp, _ = mini_grid
    assert grid.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "skip    react/mock-paid" in out           # paid without --allow-paid
    dirs = sorted(p.name for p in (tmp / "grid").iterdir())
    assert dirs == ["random-intent_none_V2_task_01_tomato_seed0", "random-intent_none_V2_task_01_tomato_seed1",
                    "react_mock_V2_task_01_tomato_seed0"]
    cfg = json.loads((tmp / "grid" / dirs[2] / "config.json").read_text())
    assert cfg["repro"]["enabled"] is True and cfg["cli_args"]["grid_config_hash"]
    # resume: nothing re-runs
    assert grid.main([str(path)]) == 0
    assert capsys.readouterr().out.count("done    ") == 3
    # a crashed / interrupted run is re-run, the old dir kept
    s = tmp / "grid" / dirs[0] / "summary.json"
    s.write_text(json.dumps({**json.loads(s.read_text()), "stop_reason": "interrupted"}))
    grid.main([str(path), "--only", "random-intent"])
    names = sorted(p.name for p in (tmp / "grid").iterdir())
    assert any(n.startswith(dirs[0] + "_stale") for n in names) and dirs[0] in names
    # dry run reports status and estimates
    grid.main([str(path), "--dry-run"])
    out = capsys.readouterr().out
    assert "frozen check: OK" in out and "done" in out and "pending" in out and "PAID" in out


def test_grid_refuses_unfrozen(mini_grid, capsys):
    path, tmp, fz = mini_grid
    fz.write_text("prompts_sha256: other\naction_reader_version: 2.1-floor-tolerant\n", encoding="utf-8")
    assert grid.main([str(path)]) == 2
    assert "does not match the frozen config" in capsys.readouterr().err
    assert not (tmp / "grid").exists()
    assert grid.main([str(path), "--only", "random-intent", "--allow-unfrozen"]) == 0


def test_config_hash_changes_with_settings(mini_grid):
    path, _, _ = mini_grid
    g = grid.load_grid(path)
    h1 = grid.combos(g)[0].config_hash
    g["max_env_steps"] = 31
    assert grid.combos(g)[0].config_hash != h1
