"""Shadow-mode trigger evaluation (oracle-side analysis; no LLM, nothing reaches an agent).

    python -m react_baseline.trigger_eval --grid-tasks --seeds 0,1,2 --steps 1000
    python -m react_baseline.trigger_eval --task V2_task_11_tomato_lettuce_onion --seeds 0 --steps 500

Runs the cheap agent once per task and seed, then replays every trigger on that same trajectory
(a fresh TKG per replay, checked before each transition is added, exactly like the TKG agent):
tkg1, tkg0 and tkg1c at default parameters, random with p = tkg1c's fire rate on that trajectory
and every_n with n = round(1 / that rate), so the baselines fire about as often. Fires count only
once the memory check runs (after the warm-up); recall counts every first occurrence in the run.
Also a small tkg1c sweep over tau and min_support. Writes runs/trigger_eval/<timestamp>/
trigger_eval.json and trigger_eval.md and prints the tables (means over seeds).
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from memory.abstraction import abstract
from memory.tkg import TKG
from memory.trigger import EveryNTrigger, RandomTrigger, TKGTrigger

from .cheap_policy import CheapAgent
from .env_adapter import REPO_ROOT
from .normal_play import GRID_TASKS
from .runner import Budgets, run
from .solutions import load_normal_play
from .trigger_metrics import evaluate

METRICS = ("fire_rate", "precision_adj", "recall_unique_adj", "recall_events_adj")
SWEEP_TAU = (0.05, 0.15, 0.3)
SWEEP_MIN_SUPPORT = (2, 3, 5)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cheap_trajectory(task: str, seed: int, steps: int, run_dir: Path):
    """Run the cheap agent; -> (abstract transitions in order, steps.jsonl rows)."""
    run(CheapAgent(seed), task, seed, Budgets(max_env_steps=steps), run_dir, {"backend": None, "model": None})
    trans, prev = [], {}
    for r in _jsonl(run_dir / "transitions.jsonl"):
        ev = r["event_records"][-1] if r["event_records"] else None
        k = abstract(r["kind"], ev, prev.get(r["episode"]))
        prev[r["episode"]] = k["outcome"]
        trans.append({**k, "id": f"t{r['global_step_start']}", "episode_key": f"ep{r['episode']}",
                      "global_step_start": r["global_step_start"], "global_step_end": r["global_step_end"],
                      "n_steps": len(r["primitive_actions"])})
    return trans, _jsonl(run_dir / "steps.jsonl")


def replay(trans: list[dict], trigger, warmup: int) -> list[dict]:
    """Memory check on every transition after ``warmup`` env steps, then add it (fresh TKG)."""
    tkg, env_steps, out = TKG(trigger.min_support), 0, []
    for t in trans:
        checked, fired = env_steps >= warmup, False
        if checked:
            fired, _ = trigger.check(t, tkg)
        tkg.add(t, None, None)
        env_steps += t["n_steps"]
        out.append({"start": t["global_step_start"], "end": t["global_step_end"], "checked": checked,
                    "fired": bool(fired)})
    return out


def score(rows: list[dict], step_rows: list[dict], normal) -> dict:
    ev = evaluate(rows, step_rows, normal)
    ev["fire_rate"] = (ev["fires"] / ev["checks"]) if ev["checks"] else None
    return ev


def eval_trajectory(trans, step_rows, normal, warmup: int, seed: int) -> dict:
    out = {}
    for name, trig in (("tkg1", TKGTrigger(1)), ("tkg0", TKGTrigger(0)), ("tkg1c", TKGTrigger(1, coarse=True))):
        out[name] = score(replay(trans, trig, warmup), step_rows, normal)
    rate = out["tkg1c"]["fire_rate"] or 0.0
    out["random"] = {**score(replay(trans, RandomTrigger(rate, seed), warmup), step_rows, normal), "p": rate}
    n = max(round(1 / rate), 1) if rate else 10 ** 9
    out["every_n"] = {**score(replay(trans, EveryNTrigger(n), warmup), step_rows, normal), "n": n}
    return out


def sweep_trajectory(trans, step_rows, normal, warmup: int) -> dict:
    return {f"tau={tau},min_support={ms}": score(replay(trans, TKGTrigger(1, ms, tau, coarse=True), warmup),
                                                 step_rows, normal)
            for tau, ms in itertools.product(SWEEP_TAU, SWEEP_MIN_SUPPORT)}


def _mean(rows: list[dict], key: str):
    xs = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(xs) if xs else None


def _fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def table(per_seed: dict[str, list[dict]], label: str = "trigger") -> str:
    lines = [f"| {label} | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    for name, rows in per_seed.items():
        lines.append(f"| {name} | " + " | ".join(_fmt(_mean(rows, m)) for m in METRICS) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", action="append")
    g.add_argument("--grid-tasks", action="store_true", help="the 3 grid tasks")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmups", default="200", help="comma-separated warm-up lengths (env steps)")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "trigger_eval"))
    a = p.parse_args(argv)
    tasks = GRID_TASKS if a.grid_tasks else a.task
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    warmups = [int(w) for w in a.warmups.split(",") if w.strip()]
    out = Path(a.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    normal_play = load_normal_play()
    result = {"steps": a.steps, "seeds": seeds, "warmups": warmups, "tasks": {}}
    md = [f"# Trigger evaluation (shadow mode, cheap-agent trajectories, {a.steps} steps, seeds {seeds})", ""]
    for task in tasks:
        normal = normal_play.get(task)
        res = result["tasks"][task] = {}
        trajs = [cheap_trajectory(task, s, a.steps, out / "runs" / f"{task}_seed{s}") for s in seeds]
        for w in warmups:
            per_trig, per_sweep = {}, {}
            for s, (trans, step_rows) in zip(seeds, trajs):
                for name, m in eval_trajectory(trans, step_rows, normal, w, s).items():
                    per_trig.setdefault(name, []).append(m)
                for name, m in sweep_trajectory(trans, step_rows, normal, w).items():
                    per_sweep.setdefault(name, []).append(m)
            res[f"warmup_{w}"] = {"triggers": per_trig, "sweep_tkg1c": per_sweep}
            for title, body in ((f"{task}, warm-up {w}", table(per_trig)),
                                (f"{task}, warm-up {w}: tkg1c sweep", table(per_sweep, "tkg1c params"))):
                md += [f"## {title}", "", body, ""]
    (out / "trigger_eval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "trigger_eval.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {out / 'trigger_eval.json'} and trigger_eval.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
