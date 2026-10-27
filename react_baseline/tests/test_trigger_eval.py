"""Oracle-side trigger evaluation: metrics, per-run summary fields, shadow-mode CLI."""
import json

from react_baseline import trigger_eval
from react_baseline.llm import MockIntentLLM
from react_baseline.runner import Budgets, run
from react_baseline.tkg_agent import TKGAgent
from react_baseline.trigger_metrics import evaluate


def _row(g, bugs):
    return {"global_step": g, "bug_ids_triggered": bugs}


def test_metrics_on_a_handcrafted_run():
    trans = [{"start": 1, "end": 2, "checked": False, "fired": False},    # warm-up: bug 7 first seen
             {"start": 3, "end": 3, "checked": True, "fired": True},      # bug 9 first seen, fired
             {"start": 4, "end": 5, "checked": True, "fired": True},      # nothing adjusted (only 4)
             {"start": 6, "end": 6, "checked": True, "fired": False}]     # bug 7 again, bug 11 first seen
    steps = [_row(2, [7]), _row(3, [9, 4]), _row(5, [4]), _row(6, [7, 11])]
    m = evaluate(trans, steps, normal=[4])
    assert m["precision_adj"] == 0.5                  # 1 of 2 fires hit an adjusted bug
    assert abs(m["recall_unique_adj"] - 1 / 3) < 1e-9  # 9 caught; 7 (warm-up) and 11 missed
    assert m["recall_events_adj"] == 0.25             # 1 of 4 adjusted occurrences
    assert (m["fires"], m["checks"], m["adjusted_bugs"]) == (2, 3, 3)
    assert evaluate(trans, steps, None)["precision_adj"] is None


def test_runner_adds_trigger_fields_for_memory_agents_only(tmp_path):
    s = run(TKGAgent(MockIntentLLM(seed=1), warmup_steps=50), "V2_task_07_tomato_steak", 1,
            Budgets(max_env_steps=300), tmp_path / "t", {"backend": "mock", "model": "mock"})
    for k in ("trigger_precision_adj", "trigger_recall_unique_adj", "trigger_recall_events_adj"):
        assert k in s
    assert s["trigger_recall_unique_adj"] is not None
    from react_baseline.agents import RandomIntentAgent
    s2 = run(RandomIntentAgent(0), "V2_task_07_tomato_steak", 0, Budgets(max_env_steps=50), tmp_path / "r",
             {"backend": None, "model": None})
    assert "trigger_precision_adj" not in s2


def test_shadow_mode_cli(tmp_path, capsys):
    assert trigger_eval.main(["--task", "V2_task_01_tomato", "--seeds", "0,1", "--steps", "300",
                              "--warmups", "0,100", "--investigate-k", "2", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    (run_dir,) = list(tmp_path.iterdir())
    data = json.loads((run_dir / "trigger_eval.json").read_text())
    task = data["tasks"]["V2_task_01_tomato"]
    assert set(task) == {"warmup_0", "warmup_100"}
    for w, res in task.items():
        assert set(res["triggers"]) == {"tkg1", "tkg0", "tkg1c", "random", "every_n"}
        assert len(res["sweep_tkg1c"]) == 18 and all(len(v) == 2 for v in res["sweep_tkg1c"].values())
        for c, r in zip(res["triggers"]["tkg1c"], res["triggers"]["random"]):
            assert r["p"] == c["fire_rate"]                     # matched fire rate
            assert c["llm_calls_projected_per_1k"] == c["fires"] * 2 * 1000 / 300
            assert len(c["fires_per_100"]) == 3 and sum(c["fires_per_100"]) == c["fires"]
        k1, k3 = (res["sweep_tkg1c"][f"tau=0.15,min_support=3,k={k}"][0] for k in (1, 3))
        assert k3["llm_calls_projected_per_1k"] == 3 * k1["llm_calls_projected_per_1k"]
    # warm-up 0 checks everything from the start; warm-up 100 never fires in the first window
    assert all(c["fires_per_100"][0] == 0 for c in task["warmup_100"]["triggers"]["tkg1"])
    assert all(c["fires_per_100"][0] > 0 for c in task["warmup_0"]["triggers"]["tkg1"])
    assert "| tkg1c |" in out and "fires per 100-step window" in out and (run_dir / "trigger_eval.md").exists()
