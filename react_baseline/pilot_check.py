"""Sanity gate for a short pilot run before spending hours on a full run.

    python -m react_baseline.pilot_check runs/<run_dir> [runs/<run_dir> ...]

The env is deterministic given the action sequence, so the checker replays
steps.jsonl to recover positions and held items (they are not logged). A run
passes only if every criterion holds; exit code 0 = all pass, 1 = any fail.

All agents: >= 5 distinct tiles, >= 1 pickup, parse error rate <= 20%, exact
replay of the primitive actions. Primitive-action agents: longest same-action
streak <= 10. Intent agents (config action_space == "intent"): longest run of
identical intents <= 10 and intent-failure rate <= 30% (a goto legitimately
repeats one raw move, so the raw streak is not a loop signal there).
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
MAX_SAME_INTENT_RUN = 10      # intent agents: no fixed-point loop over intents
MAX_INTENT_FAILURE_RATE = 0.30


def longest_run(seq) -> int:
    best = streak = 1 if seq else 0
    for a, b in zip(seq, seq[1:]):
        streak = streak + 1 if a == b else 1
        best = max(best, streak)
    return best


def intent_sequence(rows) -> list[str]:
    """One entry per executed intent (rows sharing an intent_id are one intent)."""
    out, last = [], object()
    for r in rows:
        if r.get("intent_id") != last:
            out.append(r.get("intent"))
            last = r.get("intent_id")
    return out


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

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    intent_mode = (config.get("action_space") or summary.get("action_space")) == "intent"

    per = summary.get("parse_error_rate") or 0.0
    results = [
        (len(tiles) >= MIN_DISTINCT_TILES, f"distinct tiles visited: {len(tiles)} (need >= {MIN_DISTINCT_TILES})"),
        (pickups >= MIN_PICKUPS, f"times something was picked up: {pickups} (need >= {MIN_PICKUPS})"),
    ]
    if intent_mode:
        run_len = longest_run(intent_sequence(rows))
        ifr = summary.get("intent_failure_rate") or 0.0
        results += [
            (run_len <= MAX_SAME_INTENT_RUN,
             f"longest run of identical intents: {run_len} (need <= {MAX_SAME_INTENT_RUN})"),
            (ifr <= MAX_INTENT_FAILURE_RATE,
             f"intent failure rate: {ifr:.1%} ({summary.get('intent_failures', 0)}/"
             f"{summary.get('intents_executed', 0)}) (need <= {MAX_INTENT_FAILURE_RATE:.0%})"),
        ]
    else:
        best = longest_run([r["action"] for r in rows])
        results.append((best <= MAX_SAME_ACTION_STREAK,
                        f"longest same-action streak: {best} (need <= {MAX_SAME_ACTION_STREAK})"))
    results += [
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
