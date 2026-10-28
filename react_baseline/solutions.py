"""Scripted correct-play solutions and the oracle IDs that fire during them (post hoc).

    python -m react_baseline.solutions            # run every solution, print results
    python -m react_baseline.solutions --write    # also record them in configs/normal_play_bugs.yaml

``configs/solutions/<task>.yaml`` holds one intent sequence that completes the order
through the Action reader. The IDs that fire while playing it are recorded as that task's
``scripted_solution``. They depend on the path taken, so for tasks the recipe planner has
covered (``python -m react_baseline.normal_play``, source: planner) the normal-play set is
the union over every correct-play variant, and ``--write`` only refreshes the
``scripted_solution`` entry. summary.json reports ``unique_bugs_excluding_normal_play`` next
to the existing metrics. This never changes ``ignore_bug_ids`` or what the oracle counts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLUTIONS_DIR = REPO_ROOT / "configs" / "solutions"
NORMAL_PLAY_YAML = REPO_ROOT / "configs" / "normal_play_bugs.yaml"


def solution_tasks() -> list[str]:
    return sorted(p.stem for p in SOLUTIONS_DIR.glob("*.yaml"))


def load_solution(task: str) -> list[str]:
    return list(yaml.safe_load((SOLUTIONS_DIR / f"{task}.yaml").read_text(encoding="utf-8"))["intents"])


def run_solution(task: str) -> dict:
    """Play the scripted solution offline. -> {delivered, steps, bug_ids_raw, traces}."""
    from .action_reader import ActionReader, VisibleStep
    from .env_adapter import EnvAdapter

    adapter = EnvAdapter(task)
    fired, traces, steps, delivered = set(), [], 0, False
    try:
        adapter.reset()
        state, reader = adapter.state, ActionReader()

        def step(a):
            nonlocal steps, delivered
            r = adapter.step(a)
            steps += 1
            fired.update(r.oracle["bugs"])
            delivered = delivered or r.oracle["delivery_success"]
            return VisibleStep(r.obs_text, r.summary, r.state, r.state["last_event"]["text"], r.done)

        for intent in load_solution(task):
            res = reader.execute(intent, state, step)
            traces.append(res.trace)
            state = res.final.state
            if res.done:
                break
    finally:
        adapter.close()
    return {"delivered": delivered, "steps": steps, "bug_ids_raw": sorted(int(b) for b in fired),
            "traces": traces}


def load_normal_play(path: Path | str | None = None) -> dict[str, list[int]]:
    path = Path(path or NORMAL_PLAY_YAML)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {t: list(v.get("bug_ids_raw", [])) for t, v in (data.get("tasks") or {}).items()}


def write_normal_play(results: dict[str, dict], path: Path | str | None = None):
    """Record each scripted solution's IDs; planner-covered tasks keep their union set."""
    from .normal_play import load_yaml, write_yaml
    data = load_yaml(path)
    tasks = data.setdefault("tasks", {})
    for t, r in sorted(results.items()):
        scripted = {"bug_ids_raw": r["bug_ids_raw"], "steps": r["steps"]}
        entry = tasks.get(t) or {}
        if entry.get("source") == "planner":
            entry["scripted_solution"] = scripted
        else:
            entry = {"source": "scripted_solution", "bug_ids_raw": r["bug_ids_raw"],
                     "scripted_solution": scripted}
        tasks[t] = entry
    write_yaml(data, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true", help="rewrite configs/normal_play_bugs.yaml")
    args = p.parse_args(argv)
    results, ok = {}, True
    for t in solution_tasks():
        r = run_solution(t)
        results[t] = r
        ok &= r["delivered"]
        print(f"{t}: {'DELIVERED' if r['delivered'] else 'FAILED'} in {r['steps']} steps; "
              f"bug IDs during correct play: {r['bug_ids_raw']}")
    if args.write:
        write_normal_play(results)
        print(f"wrote {NORMAL_PLAY_YAML}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
