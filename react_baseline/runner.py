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
from .repro import ReproConfig, ReproController
from .solutions import load_normal_play
from .taxonomy import taxonomy_counts
from .transitions import VisibleTransition
from .trigger_metrics import evaluate as evaluate_trigger

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
    bug_reports_discarded: int = 0
    deliveries_success: int = 0
    delivery_attempts: int = 0
    first_found: dict = field(default_factory=dict)      # bug_id -> first global_step
    curve: list = field(default_factory=list)            # discovery points
    bug_reports: list = field(default_factory=list)
    stop_reason: str = ""
    price_in_per_mtok: float | None = 0.0
    price_out_per_mtok: float | None = 0.0
    phase_calls: dict = field(default_factory=lambda: {"explore": [0, 0, 0], "repro": [0, 0, 0]})
    # phase -> [llm_calls, input_tokens, output_tokens]

    def usd_of(self, input_tokens: int, output_tokens: int) -> float | None:
        if self.price_in_per_mtok is None or self.price_out_per_mtok is None:
            return None
        return (input_tokens * self.price_in_per_mtok + output_tokens * self.price_out_per_mtok) / 1e6

    def phase_usd(self, phase: str) -> float | None:
        if self.price_in_per_mtok is None or self.price_out_per_mtok is None:
            return None
        _, i, o = self.phase_calls[phase]
        return (i * self.price_in_per_mtok + o * self.price_out_per_mtok) / 1e6

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


def decision_keys(step_rows: list[dict]) -> list[str]:
    """One key per decision: its executed intents in order (raw agents: action names)."""
    keys, seen = {}, set()
    for r in step_rows:
        d, i = r.get("decision_id"), r.get("intent_id")
        if (d, i) in seen:
            continue
        seen.add((d, i))
        keys.setdefault(d, []).append(r.get("intent") or str(r["action"]))
    return ["; ".join(v) for _, v in sorted(keys.items())]


def loop_metrics(keys: list[str]) -> dict:
    """Reporting only. Share of decisions inside runs of >= 3 identical decisions, and
    inside A,B,A,B alternations (A != B) of length >= 4."""
    n = len(keys)
    if not n:
        return {"loop_decision_fraction": None, "cycle2_decision_fraction": None}
    in_loop = [False] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and keys[j + 1] == keys[i]:
            j += 1
        if j - i + 1 >= 3:
            in_loop[i:j + 1] = [True] * (j - i + 1)
        i = j + 1
    in_cyc = [False] * n
    start = 0
    for k in range(1, n + 1):
        ok = k < n and keys[k] != keys[k - 1] and (k - start < 2 or keys[k] == keys[k - 2])
        if not ok:
            if k - start >= 4:
                in_cyc[start:k] = [True] * (k - start)
            # the next alternation may start at the previous decision
            start = k - 1 if k < n and keys[k] != keys[k - 1] else k
    return {"loop_decision_fraction": sum(in_loop) / n, "cycle2_decision_fraction": sum(in_cyc) / n}


def normal_play_excluded(task: str, found) -> dict:
    """Reporting only; never changes ignore_bug_ids or what counts as a bug."""
    normal = load_normal_play().get(task)
    if normal is None:
        return {"unique_bugs_excluding_normal_play": None, "normal_play_bug_ids": None}
    return {"unique_bugs_excluding_normal_play": len(set(found) - set(normal)),
            "normal_play_bug_ids": sorted(normal)}


def trigger_fields(trig_rows, step_rows: list[dict], task: str) -> dict:
    """trigger_precision_adj / trigger_recall_unique_adj / trigger_recall_events_adj for agents with
    a memory check (None when no check ran; absent for agents without the hook)."""
    if trig_rows is None:
        return {}
    keys = ("precision_adj", "recall_unique_adj", "recall_events_adj")
    if not any(t["checked"] for t in trig_rows):
        return {f"trigger_{k}": None for k in keys}
    ev = evaluate_trigger(trig_rows, step_rows, load_normal_play().get(task))
    return {f"trigger_{k}": ev[k] for k in keys}


def phase_split(stats: RunStats, step_rows: list[dict]) -> dict:
    out = {}
    for ph in ("explore", "repro"):
        calls, tin, tout = stats.phase_calls[ph]
        usd = stats.phase_usd(ph)
        out.update({f"steps_{ph}": sum(1 for r in step_rows if r.get("phase", "explore") == ph),
                    f"llm_calls_{ph}": calls, f"tokens_{ph}": tin + tout,
                    f"usd_{ph}": None if usd is None else round(usd, 6)})
    explore_ids = {b for r in step_rows if r.get("phase", "explore") == "explore"
                   for b in r["bug_ids_triggered"]}
    out["n_unique_bugs_explore"] = len(explore_ids)
    return out


def _as_intents(decision) -> list[Intent]:
    if getattr(decision, "intents", None) is not None:
        return [i if isinstance(i, Intent) else parse_intent(i) for i in decision.intents]
    return [primitive_intent(a) for a in decision.actions]


def run(agent, task: str, seed: int, budgets: Budgets, run_dir: Path, config: dict,
        ignore_bug_ids=(5,), progress=None, reader: ActionReader | None = None,
        ledger=None, repro: ReproConfig | None = None) -> dict:
    """Run ``agent`` on ``task`` until a budget is exhausted. Returns the summary.

    ``ledger`` (spend.Ledger): for paid runs, a reservation is appended before the first
    call and a final record after the summary is written (also after Ctrl-C or a crash).
    ``repro`` (ReproConfig): agent-driven reproduction attempts (see repro.py); their
    steps, calls and tokens count against the same budgets."""
    repro = repro or ReproConfig()
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
        "repro": repro.to_dict(),
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
    ctl = ReproController(repro, budgets.max_env_steps)
    step_rows: list[dict] = []  # kept for bug-report attribution and repro scoring
    trig_rows: list[dict] | None = [] if hasattr(agent, "on_transition") else None  # memory checks
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
            if ctl.current is not None and ctl.steps_left(stats.global_step) <= 0:
                return None  # the reproduction attempt's step allowance is used up
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
                "phase": ctl.phase, "repro_attempt_id": ctl.attempt_id,
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
            open(run_dir / "transitions.jsonl", "w", encoding="utf-8") as f_trans, \
            open(run_dir / "episodes.jsonl", "w", encoding="utf-8") as f_eps, \
            open(run_dir / "repro_attempts.jsonl", "w", encoding="utf-8") as f_repro:
        state = adapter.state
        decision_id = intent_id = -1
        stalled = 0
        ep = {"start": 1, "phase": ctl.phase}   # current episode: first global step, phase at start

        def close_episode(reason: str):
            f_eps.write(json.dumps({"episode": stats.episodes, "phase": ep["phase"],
                                    "start_global_step": ep["start"],
                                    "end_global_step": stats.global_step,
                                    "end_reason": reason}) + "\n")

        def new_episode(reason: str):
            """End the current episode (logging why) and reset the env."""
            nonlocal obs_text, state, episode_step
            close_episode(reason)
            stats.episodes += 1
            obs_text = adapter.reset()
            state = adapter.state
            episode_step = 0
            ep["start"], ep["phase"] = stats.global_step + 1, ctl.phase

        def episode_has_steps() -> bool:
            return stats.global_step >= ep["start"]

        def finish_attempt(reason: str, reset: bool = True):
            row = ctl.finish(reason, stats.global_step, step_rows)
            f_repro.write(json.dumps(row) + "\n")
            f_repro.flush()
            if reset and episode_has_steps():
                new_episode("repro_end")
            else:
                ep["phase"] = ctl.phase
            hook = getattr(agent, "on_repro_end", None)
            if hook:
                hook()

        try:
            while budget_left():
                if ctl.current is not None:
                    tick = getattr(agent, "on_repro_tick", None)
                    if tick:
                        tick(ctl.steps_left(stats.global_step))
                phase = ctl.phase
                decision = agent.act(obs_text, state)
                decision_id += 1
                stats.decisions += 1
                intents = _as_intents(decision)
                call_id, call_row, rec = None, None, decision.llm_call
                if rec is not None:
                    call_id = stats.llm_calls
                    stats.llm_calls += 1
                    stats.input_tokens += rec["input_tokens"]
                    stats.output_tokens += rec["output_tokens"]
                    pc = stats.phase_calls[phase]
                    pc[0] += 1
                    pc[1] += rec["input_tokens"]
                    pc[2] += rec["output_tokens"]
                    if ctl.current is not None:
                        ctl.current["llm_calls"] += 1
                        ctl.current["input_tokens"] += rec["input_tokens"]
                        ctl.current["output_tokens"] += rec["output_tokens"]
                        ctl.current["usd"] += (stats.usd_of(rec["input_tokens"], rec["output_tokens"]) or 0.0)
                    stats.latencies.append(rec["latency_s"])
                    if rec["parse_error"]:
                        stats.parse_errors += 1
                    if rec.get("bug_report_discarded"):
                        stats.bug_reports_discarded += 1
                    call_row = {"llm_call_id": call_id, **llm_meta,
                                "global_step": stats.global_step, "episode": stats.episodes, **rec,
                                "decision_id": decision_id,
                                "parsed_intents": [i.text for i in intents],
                                "phase": phase, "repro_attempt_id": ctl.attempt_id,
                                "parsed_reproduced": bool(rec.get("parsed_reproduced"))}
                    if rec["parsed_bug_report"]:
                        stats.bug_reports.append({
                            "llm_call_id": call_id, "global_step": stats.global_step,
                            "episode": stats.episodes, "report": rec["parsed_bug_report"],
                            "decision_id": decision_id, "phase": phase,
                            "repro_attempt_id": ctl.attempt_id})

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
                            extra = {}
                            hook = getattr(agent, "on_transition", None)
                            if hook:   # agent-visible data only (memory agents build their TKG here)
                                extra = hook(r, prev_state, state, {
                                    "run": f"{task}/seed{seed}", "episode": stats.episodes,
                                    "global_step_start": start, "global_step_end": stats.global_step,
                                    "decision_id": decision_id, "intent_id": intent_id,
                                    "phase": phase}) or {}
                                dev = extra.get("deviation")
                                counted = bool(dev) and dev.get("counted", True)
                                trig_rows.append({"start": start, "end": stats.global_step, "checked": counted,
                                                  "fired": counted and bool(dev.get("fire"))})
                            stats.intent_failures += r.error is not None
                            f_trans.write(json.dumps(VisibleTransition(
                                prev_state=prev_state, intent=r.intent,
                                primitive_actions=r.primitive_actions, events=r.events,
                                new_state=state, decision_id=decision_id, intent_id=intent_id,
                                episode=stats.episodes, global_step_start=start,
                                global_step_end=stats.global_step, kind=r.kind,
                                event_records=r.event_records, stop_reason=r.stop_reason,
                                error=r.error, note=r.note, episode_ended=r.done,
                                coerced=r.coerced, phase=phase,
                                repro_attempt_id=ctl.attempt_id, actor=extra.get("actor"),
                                deviation=extra.get("deviation"),
                                trigger_reason=extra.get("trigger_reason")).to_row()) + "\n")
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
                if progress and rec is not None:
                    progress(stats, decision)

                executed = [a for r in results for a in r.primitive_actions]
                if executed:
                    agent.observe(executed, obs_summary, results)
                if done:
                    new_episode("done")
                    agent.on_episode_reset()

                # -- reproduction protocol (agent-side information only) --------
                if ctl.current is not None:
                    claimed = bool(rec and rec.get("parsed_reproduced"))
                    if claimed:
                        ctl.current["agent_claimed_reproduced"] = True
                    if claimed or ctl.steps_left(stats.global_step) <= 0:
                        finish_attempt("claimed" if claimed else "steps")
                elif repro.enabled and rec is not None and rec["parsed_bug_report"]:
                    report = rec["parsed_bug_report"]
                    if ctl.consider(report, stats.global_step) is None:
                        prev_rows = [r for r in step_rows if r["decision_id"] == decision_id - 1]
                        strict = sorted({b for r in prev_rows for b in r["bug_ids_triggered"]})
                        if episode_has_steps():
                            new_episode("repro_start")
                        ctl.start(report, {"decision_id": decision_id, "llm_call_id": call_id,
                                           "global_step": stats.global_step}, strict,
                                  stats.global_step)
                        ep["phase"] = ctl.phase
                        hook = getattr(agent, "on_repro_start", None)
                        if hook:
                            hook(report, repro.steps)

                # the reader always runs >= 1 step unless a step allowance is used up;
                # never spin on decisions that execute nothing
                stalled = 0 if executed else stalled + 1
                if stalled >= 3:
                    stats.stop_reason = stats.stop_reason or "stalled"
                    break
        except KeyboardInterrupt:
            stats.stop_reason = "interrupted"
        except Exception as exc:  # still write bug_reports + summary, then re-raise
            stats.stop_reason = f"error: {type(exc).__name__}: {exc}"
            error = exc
        finally:
            adapter.close()
        if ctl.current is not None:  # the run ended inside an attempt
            finish_attempt(stats.stop_reason or "budget", reset=False)
        if episode_has_steps():
            close_episode(stats.stop_reason or "budget")

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
        "bug_reports_discarded": stats.bug_reports_discarded,   # non-reports ("no defect confirmed"), not counted
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
        # explore vs agent-driven reproduction
        **phase_split(stats, step_rows),
        **ctl.summary(),
        # stuck-behaviour metrics (reporting only)
        **loop_metrics(decision_keys(step_rows)),
        # post hoc: IDs that also fire during scripted correct play (configs/normal_play_bugs.yaml)
        **normal_play_excluded(task, stats.first_found),
        # oracle-side evaluation of the memory check (analysis only; see trigger_metrics.py)
        **trigger_fields(trig_rows, step_rows, task),
    }
    fields = getattr(agent, "summary_fields", None)   # e.g. the TKG arm's escalations, tkg size
    if fields:
        summary.update({k: v for k, v in fields().items() if k not in summary})
    artifacts = getattr(agent, "write_artifacts", None)   # e.g. tkg.json
    if artifacts:
        artifacts(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if ledger is not None:
        status = ("error" if stats.stop_reason.startswith("error") else
                  "interrupted" if stats.stop_reason == "interrupted" else "completed")
        ledger.finalize(run_dir, summary["usd_cost"], stats.input_tokens, stats.output_tokens,
                        status, stats.stop_reason)
    if error is not None:
        raise error
    return summary
