"""Budgeted run loop with auto-reset and research logging.

The runner is the only component that sees the hidden oracle (bug predicate
IDs). It writes them to steps.jsonl but never passes them to the agent.

Every decision goes through the shared Action reader: an agent returns intents
(or raw actions, which become one-step intents), the reader turns them into
button presses through ``step_fn``, and ``step_fn`` hands back only the
agent-visible ``VisibleStep``.
"""
from __future__ import annotations

import json
import random
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .action_reader import ActionReader, Intent, VisibleStep, parse_intent, primitive_intent
from .env_adapter import EnvAdapter, REPO_ROOT, SAGE_ROOT
from .taxonomy import taxonomy_counts
from .transitions import VisibleTransition

BUG_REPORT_WINDOW = 3  # bug_reports.jsonl pairs each report with oracle bugs within ±3 steps


@dataclass
class Budgets:
    max_env_steps: int = 2000
    max_llm_calls: int | None = None
    max_tokens: int | None = None
    max_usd: float | None = None


@dataclass
class RunStats:
    global_step: int = 0
    episodes: int = 1
    llm_calls: int = 0
    decisions: int = 0
    intents_executed: int = 0
    intent_failures: int = 0
    intents_coerced: int = 0
    latencies: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    parse_errors: int = 0
    deliveries_success: int = 0
    delivery_attempts: int = 0
    first_found: dict = field(default_factory=dict)      # bug_id -> first global_step
    curve: list = field(default_factory=list)            # discovery points
    bug_reports: list = field(default_factory=list)
    stop_reason: str = ""
    price_in_per_mtok: float | None = 0.0
    price_out_per_mtok: float | None = 0.0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def usd(self) -> float | None:
        if self.price_in_per_mtok is None or self.price_out_per_mtok is None:
            return None
        return (self.input_tokens * self.price_in_per_mtok
                + self.output_tokens * self.price_out_per_mtok) / 1e6


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


def attribute_reports(reports: list[dict], step_rows: list[dict],
                      window: int = BUG_REPORT_WINDOW) -> list[dict]:
    """Attach oracle evidence to each bug report.

    Lenient (legacy): ``bug_ids_in_window`` = oracle IDs within ±``window`` steps of
    the report's ``global_step`` (the steps done before the reporting call).
    Strict: a report made at decision d is about the observation produced by
    decision d-1, so ``bug_ids_strict`` = oracle IDs that fired during the env
    steps executed by decision d-1 (``evidence_steps``).
    """
    by_decision: dict[int, list[dict]] = {}
    for r in step_rows:
        by_decision.setdefault(r.get("decision_id"), []).append(r)
    out = []
    for rep in reports:
        g = rep["global_step"]
        lo, hi = g - window, g + window
        nearby = sorted({b for r in step_rows if lo <= r["global_step"] <= hi
                         for b in r["bug_ids_triggered"]})
        d = rep.get("decision_id")
        prev = by_decision.get(d - 1, []) if d is not None else []
        out.append({**rep, "window": [lo, hi], "bug_ids_in_window": nearby,
                    "bug_ids_strict": sorted({b for r in prev for b in r["bug_ids_triggered"]}),
                    "evidence_steps": [r["global_step"] for r in prev]})
    return out


def _as_intents(decision) -> list[Intent]:
    if getattr(decision, "intents", None) is not None:
        return [i if isinstance(i, Intent) else parse_intent(i) for i in decision.intents]
    return [primitive_intent(a) for a in decision.actions]


def run(agent, task: str, seed: int, budgets: Budgets, run_dir: Path, config: dict,
        ignore_bug_ids=(5,), progress=None, reader: ActionReader | None = None,
        ledger=None) -> dict:
    """Run ``agent`` on ``task`` until a budget is exhausted. Returns the summary.

    ``ledger`` (spend.Ledger): for paid runs, a reservation is appended before the first
    call and a final record after the summary is written (also after Ctrl-C or a crash)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ignore = set(int(b) for b in ignore_bug_ids)
    adapter = EnvAdapter(task)
    seed_everything(seed, adapter)
    reader = reader or ActionReader()
    action_space = getattr(agent, "action_space", "primitive")

    config = dict(config)
    config.update({
        "task": task, "seed": seed, "agent": agent.name, "action_space": action_space,
        "max_macro_len": reader.max_macro_len,
        "ignore_bug_ids": sorted(ignore),
        "budgets": vars(budgets),
        "git_commit": git_commit(REPO_ROOT), "git_dirty": git_dirty(REPO_ROOT),
        "sage_commit": git_commit(SAGE_ROOT),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    stats = RunStats()
    prof = config.get("model_profile") or {}
    if prof:
        stats.price_in_per_mtok = prof.get("price_in_per_mtok")
        stats.price_out_per_mtok = prof.get("price_out_per_mtok")
    if budgets.max_usd is not None and stats.usd is None:
        raise ValueError("--max-usd needs a model profile with known prices")
    if ledger is not None:
        if budgets.max_usd is None:
            raise ValueError("a ledgered (paid) run needs max_usd")
        ledger.reserve(run_dir, prof.get("name"), prof.get("model"), budgets.max_usd)
    error = None
    step_rows: list[dict] = []  # kept for bug-report attribution
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
        if budgets.max_usd is not None and stats.usd >= budgets.max_usd:
            stats.stop_reason = "max_usd"
            return False
        return True

    def make_step_fn(f_steps, decision_id: int, intent_id: int, intent: Intent, call_id):
        k = 0  # intent_step_index

        def step_fn(action: int) -> VisibleStep | None:
            nonlocal k, episode_step
            if stats.global_step >= budgets.max_env_steps:
                return None
            res = adapter.step(action)
            stats.global_step += 1
            episode_step = res.episode_step
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
                "decision_id": decision_id, "intent_id": intent_id,
                "intent": intent.text, "intent_step_index": k,
            }
            k += 1
            step_rows.append(row)
            f_steps.write(json.dumps(row) + "\n")
            # agent-visible only: the oracle stays in this function
            event = (res.state.get("last_event") or {}).get("text") or ""
            return VisibleStep(res.obs_text, res.summary, res.state, event, res.done)

        return step_fn

    with open(run_dir / "steps.jsonl", "w", encoding="utf-8") as f_steps, \
            open(run_dir / "llm_calls.jsonl", "w", encoding="utf-8") as f_calls, \
            open(run_dir / "transitions.jsonl", "w", encoding="utf-8") as f_trans:
        state = adapter.state
        decision_id = intent_id = -1
        try:
            while budget_left():
                decision = agent.act(obs_text, state)
                decision_id += 1
                stats.decisions += 1
                intents = _as_intents(decision)
                call_id, call_row = None, None
                if decision.llm_call is not None:
                    call_id = stats.llm_calls
                    stats.llm_calls += 1
                    rec = decision.llm_call
                    stats.input_tokens += rec["input_tokens"]
                    stats.output_tokens += rec["output_tokens"]
                    stats.latencies.append(rec["latency_s"])
                    if rec["parse_error"]:
                        stats.parse_errors += 1
                    call_row = {"llm_call_id": call_id, **llm_meta,
                                "global_step": stats.global_step, "episode": stats.episodes, **rec,
                                "decision_id": decision_id,
                                "parsed_intents": [i.text for i in intents]}
                    if rec["parsed_bug_report"]:
                        stats.bug_reports.append({
                            "llm_call_id": call_id, "global_step": stats.global_step,
                            "episode": stats.episodes, "report": rec["parsed_bug_report"],
                            "decision_id": decision_id})

                results, done, obs_summary = [], False, ""
                try:
                    for intent in intents:
                        if stats.global_step >= budgets.max_env_steps:
                            break
                        intent_id += 1
                        prev_state, start = state, stats.global_step + 1
                        step_fn = make_step_fn(f_steps, decision_id, intent_id, intent, call_id)
                        r = reader.execute(intent, state, step_fn)
                        results.append(r)
                        if r.final is not None:
                            obs_text, state, obs_summary = r.final.obs_text, r.final.state, r.final.summary
                        if r.n_steps:
                            stats.intents_executed += 1
                            stats.intent_failures += r.error is not None
                            f_trans.write(json.dumps(VisibleTransition(
                                prev_state=prev_state, intent=r.intent,
                                primitive_actions=r.primitive_actions, events=r.events,
                                new_state=state, decision_id=decision_id, intent_id=intent_id,
                                episode=stats.episodes, global_step_start=start,
                                global_step_end=stats.global_step, kind=r.kind,
                                event_records=r.event_records, stop_reason=r.stop_reason,
                                error=r.error, note=r.note, episode_ended=r.done,
                                coerced=r.coerced).to_row()) + "\n")
                            stats.intents_coerced += r.coerced is not None
                        if r.done:
                            done = True
                            break
                        if r.stop_reason == "budget":
                            break
                finally:
                    if call_row is not None:
                        call_row["executor_result"] = [r.to_dict() for r in results]
                        f_calls.write(json.dumps(call_row) + "\n")
                        f_calls.flush()
                decision.results = results
                if progress and decision.llm_call is not None:
                    progress(stats, decision)

                executed = [a for r in results for a in r.primitive_actions]
                if executed:
                    agent.observe(executed, obs_summary, results)
                if done:
                    stats.episodes += 1
                    obs_text = adapter.reset()
                    state = adapter.state
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

    report_rows = attribute_reports(stats.bug_reports, step_rows)
    with open(run_dir / "bug_reports.jsonl", "w", encoding="utf-8") as f:
        for row in report_rows:
            f.write(json.dumps(row) + "\n")
    strict_ids = sorted({b for r in report_rows for b in r["bug_ids_strict"]})

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
        # v2 additions
        "action_space": action_space,
        "usd_cost": None if stats.usd is None else round(stats.usd, 6),
        "decisions": stats.decisions,
        "llm_calls_per_decision": (stats.llm_calls / stats.decisions) if stats.decisions else None,
        "llm_calls_per_env_step": (stats.llm_calls / stats.global_step) if stats.global_step else None,
        "llm_latency_mean_s": round(statistics.mean(stats.latencies), 4) if stats.latencies else None,
        "llm_latency_median_s": round(statistics.median(stats.latencies), 4) if stats.latencies else None,
        "intents_executed": stats.intents_executed,
        "intent_failures": stats.intent_failures,
        "intent_failure_rate": (stats.intent_failures / stats.intents_executed) if stats.intents_executed else None,
        "intents_coerced": stats.intents_coerced,
        "intent_coercion_rate": (stats.intents_coerced / stats.intents_executed) if stats.intents_executed else None,
        "reports_strict_match": sum(1 for r in report_rows if r["bug_ids_strict"]),
        "reports_lenient_match": sum(1 for r in report_rows if r["bug_ids_in_window"]),
        "unique_bugs_reported_strict": len(strict_ids),
        "unique_bug_ids_reported_strict": strict_ids,
        # draft taxonomy: reporting only, never used to filter
        **taxonomy_counts(stats.first_found),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if ledger is not None:
        status = ("error" if stats.stop_reason.startswith("error") else
                  "interrupted" if stats.stop_reason == "interrupted" else "completed")
        ledger.finalize(run_dir, summary["usd_cost"], stats.input_tokens, stats.output_tokens,
                        status, stats.stop_reason)
    if error is not None:
        raise error
    return summary
