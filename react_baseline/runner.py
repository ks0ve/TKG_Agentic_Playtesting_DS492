"""Budgeted run loop with auto-reset and research logging.

The runner is the only component that sees the hidden oracle (bug predicate
IDs). It writes them to steps.jsonl but never passes them to the agent.
"""
from __future__ import annotations

import json
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .env_adapter import EnvAdapter, REPO_ROOT, SAGE_ROOT

BUG_REPORT_WINDOW = 3  # bug_reports.jsonl pairs each report with oracle bugs within ±3 steps


@dataclass
class Budgets:
    max_env_steps: int = 2000
    max_llm_calls: int | None = None
    max_tokens: int | None = None


@dataclass
class RunStats:
    global_step: int = 0
    episodes: int = 1
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    parse_errors: int = 0
    deliveries_success: int = 0
    delivery_attempts: int = 0
    first_found: dict = field(default_factory=dict)      # bug_id -> first global_step
    curve: list = field(default_factory=list)            # discovery points
    bug_reports: list = field(default_factory=list)
    stop_reason: str = ""

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_dirty(path: Path) -> bool | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def seed_everything(seed: int, adapter: EnvAdapter):
    random.seed(seed)
    np.random.seed(seed)
    adapter.env.action_space.seed(seed)
    # The env itself has no RNG beyond TaskManager's random.seed(42) on reset,
    # so env dynamics are deterministic given the action sequence.


def make_run_dir(out_root: Path, agent: str, task: str, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root) / f"{stamp}_{agent}_{task}_seed{seed}"
    suffix = 1
    while run_dir.exists():
        run_dir = Path(out_root) / f"{stamp}_{agent}_{task}_seed{seed}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def run(agent, task: str, seed: int, budgets: Budgets, run_dir: Path, config: dict,
        ignore_bug_ids=(5,), progress=None) -> dict:
    """Run ``agent`` on ``task`` until a budget is exhausted. Returns the summary."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ignore = set(int(b) for b in ignore_bug_ids)
    adapter = EnvAdapter(task)
    seed_everything(seed, adapter)

    config = dict(config)
    config.update({
        "task": task, "seed": seed, "agent": agent.name,
        "ignore_bug_ids": sorted(ignore),
        "budgets": vars(budgets),
        "git_commit": git_commit(REPO_ROOT), "git_dirty": git_dirty(REPO_ROOT),
        "sage_commit": git_commit(SAGE_ROOT),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    stats = RunStats()
    error = None
    step_rows: list[dict] = []  # kept for the bug-report ±3 window
    t0 = time.perf_counter()
    obs_text = adapter.reset()
    episode_step = 0
    llm_meta = {"backend": config.get("backend"), "model": config.get("model")}

    def budget_left() -> bool:
        if stats.global_step >= budgets.max_env_steps:
            stats.stop_reason = "max_env_steps"
            return False
        if budgets.max_llm_calls is not None and stats.llm_calls >= budgets.max_llm_calls:
            stats.stop_reason = "max_llm_calls"
            return False
        if budgets.max_tokens is not None and stats.tokens >= budgets.max_tokens:
            stats.stop_reason = "max_tokens"
            return False
        return True

    with open(run_dir / "steps.jsonl", "w", encoding="utf-8") as f_steps, \
            open(run_dir / "llm_calls.jsonl", "w", encoding="utf-8") as f_calls:
        try:
            while budget_left():
                decision = agent.act(obs_text)
                call_id = None
                if decision.llm_call is not None:
                    call_id = stats.llm_calls
                    stats.llm_calls += 1
                    rec = decision.llm_call
                    stats.input_tokens += rec["input_tokens"]
                    stats.output_tokens += rec["output_tokens"]
                    if rec["parse_error"]:
                        stats.parse_errors += 1
                    row = {"llm_call_id": call_id, **llm_meta,
                           "global_step": stats.global_step, "episode": stats.episodes, **rec}
                    f_calls.write(json.dumps(row) + "\n")
                    f_calls.flush()
                    if rec["parsed_bug_report"]:
                        stats.bug_reports.append({
                            "llm_call_id": call_id, "global_step": stats.global_step,
                            "episode": stats.episodes, "report": rec["parsed_bug_report"]})

                executed, done, obs_summary = [], False, ""
                for action in decision.actions:
                    if stats.global_step >= budgets.max_env_steps:
                        break
                    res = adapter.step(action)
                    stats.global_step += 1
                    episode_step = res.episode_step
                    executed.append(action)
                    obs_summary, done = res.summary, res.done

                    bugs = sorted(set(b for b in res.oracle["bugs"] if b not in ignore))
                    new = [b for b in bugs if b not in stats.first_found]
                    for b in new:
                        stats.first_found[b] = stats.global_step
                    if new:
                        stats.curve.append({"global_step": stats.global_step,
                                            "cumulative_tokens": stats.tokens,
                                            "unique_bugs": len(stats.first_found)})
                    if res.oracle["event_type"] == "deliver":
                        stats.delivery_attempts += 1
                    if res.oracle["delivery_success"]:
                        stats.deliveries_success += 1
                    row = {
                        "global_step": stats.global_step, "episode": stats.episodes,
                        "episode_step": episode_step, "action": int(action),
                        "reward": res.reward, "done": res.done,
                        "bug_ids_triggered": bugs, "new_unique_bug_ids": new,
                        "bug_ids_raw": sorted(set(res.oracle["bugs"])),
                        "delivery_success": res.oracle["delivery_success"],
                        "llm_call_id": call_id,
                    }
                    step_rows.append(row)
                    f_steps.write(json.dumps(row) + "\n")
                    obs_text = res.obs_text
                    if done:
                        break
                if progress and decision.llm_call is not None:
                    progress(stats, decision)

                if executed:
                    agent.observe(executed, obs_summary)
                if done:
                    stats.episodes += 1
                    obs_text = adapter.reset()
                    episode_step = 0
                    agent.on_episode_reset()
                elif not executed:
                    break  # step budget hit before any action ran
        except KeyboardInterrupt:
            stats.stop_reason = "interrupted"
        except Exception as exc:  # still write bug_reports + summary, then re-raise
            stats.stop_reason = f"error: {type(exc).__name__}: {exc}"
            error = exc
        finally:
            adapter.close()

    # bug reports with oracle bugs within ±window steps of the report
    with open(run_dir / "bug_reports.jsonl", "w", encoding="utf-8") as f:
        for rep in stats.bug_reports:
            g = rep["global_step"]
            lo, hi = g - BUG_REPORT_WINDOW, g + BUG_REPORT_WINDOW
            nearby = sorted({b for r in step_rows if lo <= r["global_step"] <= hi
                             for b in r["bug_ids_triggered"]})
            f.write(json.dumps({**rep, "window": [lo, hi], "bug_ids_in_window": nearby}) + "\n")

    wall = time.perf_counter() - t0
    # an episode counter that was bumped by a final reset with no steps does not count
    episodes = stats.episodes if (step_rows and step_rows[-1]["episode"] == stats.episodes) \
        else stats.episodes - 1
    curve = list(stats.curve)
    curve.append({"global_step": stats.global_step, "cumulative_tokens": stats.tokens,
                  "unique_bugs": len(stats.first_found)})
    summary = {
        "agent": agent.name, "task": task, "seed": seed,
        "backend": config.get("backend"), "model": config.get("model"),
        "stop_reason": stats.stop_reason,
        "parse_error_rate": (stats.parse_errors / stats.llm_calls) if stats.llm_calls else None,
        "parse_errors": stats.parse_errors,
        "total_env_steps": stats.global_step,
        "episodes": max(episodes, 0),
        "llm_calls": stats.llm_calls,
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "total_tokens": stats.tokens,
        "wall_time_s": round(wall, 2),
        "unique_bug_ids": sorted(stats.first_found),
        "n_unique_bugs": len(stats.first_found),
        "first_found_step": {str(k): v for k, v in sorted(stats.first_found.items())},
        "discovery_curve": curve,
        "deliveries_success": stats.deliveries_success,
        "delivery_attempts": stats.delivery_attempts,
        "agent_bug_reports": len(stats.bug_reports),
        "ignore_bug_ids": sorted(ignore),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if error is not None:
        raise error
    return summary
