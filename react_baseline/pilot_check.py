"""Sanity gate for a short pilot run before spending hours on a full run.

    python -m react_baseline.pilot_check runs/<run_dir> [runs/<run_dir> ...]

The env is deterministic given the action sequence, so the checker replays
steps.jsonl to recover positions and held items (they are not logged). A run
passes only if every criterion holds; exit code 0 = all pass, 1 = any fail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .env_adapter import EnvAdapter

MIN_DISTINCT_TILES = 5        # left the start area
MIN_PICKUPS = 1               # picked something up at least once
MAX_SAME_ACTION_STREAK = 10   # no fixed-point loop (EXPERIMENT_LOG.md, run #1)
MAX_PARSE_ERROR_RATE = 0.20


def check(run_dir: Path) -> tuple[bool, list[str]]:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
    if not rows:
        return False, ["no steps logged"]

    adapter = EnvAdapter(summary["task"])
    tiles, pickups, mismatch = set(), 0, 0
    try:
        adapter.reset()
        agent = adapter.env.unwrapped.agent[0]
        tiles.add((agent.x, agent.y))
        prev_holding = None
        for r in rows:
            res = adapter.step(r["action"])
            if res.done != r["done"] or res.episode_step != r["episode_step"]:
                mismatch += 1
            agent = adapter.env.unwrapped.agent[0]
            tiles.add((agent.x, agent.y))
            if agent.holding is not None and agent.holding is not prev_holding:
                pickups += 1
            prev_holding = agent.holding
            if res.done:
                adapter.reset()
                prev_holding = None
    finally:
        adapter.close()

    streak = best = 1
    for a, b in zip(rows, rows[1:]):
        streak = streak + 1 if a["action"] == b["action"] else 1
        best = max(best, streak)

    per = summary.get("parse_error_rate") or 0.0
    results = [
        (len(tiles) >= MIN_DISTINCT_TILES, f"distinct tiles visited: {len(tiles)} (need >= {MIN_DISTINCT_TILES})"),
        (pickups >= MIN_PICKUPS, f"times something was picked up: {pickups} (need >= {MIN_PICKUPS})"),
        (best <= MAX_SAME_ACTION_STREAK, f"longest same-action streak: {best} (need <= {MAX_SAME_ACTION_STREAK})"),
        (per <= MAX_PARSE_ERROR_RATE, f"parse error rate: {per:.1%} (need <= {MAX_PARSE_ERROR_RATE:.0%})"),
        (mismatch == 0, f"replay mismatches vs steps.jsonl: {mismatch} (need 0)"),
    ]
    lines = [f"[{'PASS' if ok else 'FAIL'}] {msg}" for ok, msg in results]
    lines.append(f"(info) unique bug IDs: {summary['unique_bug_ids']}, deliveries: "
                 f"{summary['deliveries_success']}, agent bug reports: {summary['agent_bug_reports']}")
    return all(ok for ok, _ in results), lines


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    all_ok = True
    for d in argv:
        ok, lines = check(Path(d))
        all_ok &= ok
        print(f"== {d}: {'PILOT PASSED' if ok else 'PILOT FAILED'}")
        print("\n".join("   " + line for line in lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
