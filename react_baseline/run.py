"""CLI: python -m react_baseline.run --agent {react,react-raw,random,random-intent,cheap,tkg} --task <id> ..."""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from . import pilot_check, spend
from .action_reader import DEFAULT_MAX_MACRO_LEN, ActionReader
from .agents import ContextOverflowError, RandomAgent, RandomIntentAgent, ReActAgent
from .cheap_policy import CheapAgent
from .tkg_agent import TKGAgent
from .repro import ReproConfig
from memory.trigger import make_trigger
from .env_adapter import list_tasks
from .llm import BACKENDS, MODELS_YAML, LLMSetupError, make_llm_from_profile, resolve_profile
from .runner import Budgets, make_run_dir, run

AGENTS = {
    "react": "ReAct, intent action space (headline baseline)",
    "react-raw": "ReAct, raw moves 0-4 (MVP ablation)",
    "random": "uniform over the 5 raw actions",
    "random-intent": "uniform over currently valid intents",
    "cheap": "no-LLM TKG novelty policy (intents; RQ1 reference)",
    "tkg": "TKG arm: cheap policy + memory check; the LLM investigates deviations",
}
LLM_AGENTS = ("react", "react-raw", "tkg")
PILOT_STEPS = 50
PAID_MAX_USD_CEILING = 2.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ReAct / random bug-hunting baseline on Overcooked Plus")
    p.add_argument("--agent", choices=list(AGENTS), required=True,
                   help="; ".join(f"{k}: {v}" for k, v in AGENTS.items()))
    p.add_argument("--task", default="V2_task_01_tomato")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds, e.g. 0,1,2: one run (and run dir) per seed; "
                        "overrides --seed")
    p.add_argument("--max-env-steps", type=int, default=2000)
    p.add_argument("--max-llm-calls", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None, help="input + output tokens")
    p.add_argument("--max-usd", type=float, default=None,
                   help="budget: USD at the profile's prices (checked before each call, so a "
                        "run can overshoot by at most one call)")
    p.add_argument("--allow-paid", action="store_true",
                   help="required for any profile with a price > 0 (spends real money)")
    p.add_argument("--max-usd-override", action="store_true",
                   help=f"allow --max-usd above ${PAID_MAX_USD_CEILING:.0f} for one paid run")
    p.add_argument("--ledger", default=None, help="spend ledger path (default spend/ledger.jsonl)")
    p.add_argument("--budget-file", default=None, help="budget YAML (default configs/budget.yaml)")
    p.add_argument("--model-profile", default=None,
                   help=f"named profile in {MODELS_YAML.name} (backend, model, prices, context "
                        "window). Without it, the --llm backend's default profile is used.")
    p.add_argument("--llm", choices=list(BACKENDS), default=None,
                   help="LLM backend (default: ollama, free/local, unless --model-profile says "
                        "otherwise). The paid API only runs with an anthropic profile.")
    p.add_argument("--model", default=None, help="override the profile's model id")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: the profile's (0.7 for the shipped profiles; at 0.0 the agent "
                        "can lock into a repeating reply, see EXPERIMENT_LOG.md)")
    p.add_argument("--max-output-tokens", type=int, default=None)
    p.add_argument("--history", type=int, default=10)
    p.add_argument("--actions-per-call", type=int, default=1,
                   help="react-raw only: raw actions allowed per call")
    p.add_argument("--intents-per-call", type=int, default=1,
                   help="react only: intents allowed per call (default 1 = consulted at every "
                        "decision point)")
    p.add_argument("--max-macro-len", type=int, default=DEFAULT_MAX_MACRO_LEN,
                   help="Action reader: max env steps one intent may take")
    p.add_argument("--num-ctx", type=int, default=None,
                   help="context window (default: the profile's; for Ollama also num_ctx)")
    p.add_argument("--ollama-host", default=None, help="override the profile's base_url")
    p.add_argument("--ignore-bug-ids", default="5",
                   help="comma-separated oracle bug IDs to drop from logs/metrics "
                        "(default 5: fires on every completed chop on single-knife maps; see README)")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-dir", default=None,
                   help="exact run directory (single seed; used by the grid driver)")
    p.add_argument("--grid-config-hash", default=None, help=argparse.SUPPRESS)
    p.add_argument("--pilot", action="store_true",
                   help=f"{PILOT_STEPS}-step pilot into runs/pilot (unless --out-dir is given), "
                        "then run pilot_check on each run; exit code 1 if any check fails")
    g = p.add_argument_group("agent-driven reproduction (counts against the run's budgets)")
    g.add_argument("--repro", action="store_true",
                   help="after a new agent bug report, reset and let the agent try to reproduce it")
    g.add_argument("--repro-steps", type=int, default=ReproConfig.steps, help="env steps per attempt")
    g.add_argument("--max-repro-attempts", type=int, default=ReproConfig.max_attempts)
    g.add_argument("--max-repro-fraction", type=float, default=ReproConfig.max_fraction,
                   help="max share of --max-env-steps spent on attempts")
    g.add_argument("--repro-dedup-jaccard", type=float, default=ReproConfig.dedup_jaccard,
                   help="skip reports whose token Jaccard with an attempted report is >= this")
    t = p.add_argument_group("TKG arm (--agent tkg)")
    t.add_argument("--trigger", choices=["tkg1", "tkg0", "tkg1c", "random", "every_n"], default="tkg1",
                   help="memory check: tkg1 (temporal context, backoff), tkg0 (current situation only), "
                        "tkg1c (tkg1 with a coarse (intent, target kind) fallback), "
                        "random (p), every_n (n)")
    t.add_argument("--trigger-p", type=float, default=0.1, help="random trigger probability")
    t.add_argument("--trigger-n", type=int, default=10, help="every_n trigger period")
    t.add_argument("--warmup-steps", type=int, default=200,
                   help="env steps of cheap-only memory building first (count in the budget); 0 = "
                        "memory starts empty and a novel situation escalates on first sight")
    t.add_argument("--investigate-k", type=int, default=3, help="LLM decisions per escalation")
    t.add_argument("--max-consecutive-llm", type=int, default=6,
                   help="cap on LLM decisions in a row, across fires that restart k (then cap_release)")
    t.add_argument("--tau", type=float, default=0.15, help="surprise threshold P(outcome) < tau")
    t.add_argument("--min-support", type=int, default=3,
                   help="observations needed before a context is judged (ctx1 backs off below it)")
    t.add_argument("--repro-path", action=argparse.BooleanOptionalAction, default=True,
                   help="add the recorded path from the TKG to reproduction prompts")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    args.repro_config = ReproConfig(args.repro, args.repro_steps, args.max_repro_attempts,
                                    args.max_repro_fraction, args.repro_dedup_jaccard)
    if args.task not in list_tasks():
        p.error(f"unknown task {args.task!r}; see external/SAGE/scenarios/overcooked_plus/tasks.json")
    if args.actions_per_call < 1 or args.intents_per_call < 1:
        p.error("--actions-per-call and --intents-per-call must be >= 1")
    if args.warmup_steps < 0:
        p.error("--warmup-steps must be >= 0 (0 = memory building and testing interleave from step 1)")
    if args.investigate_k < 1 or args.max_consecutive_llm < 1 or args.min_support < 1:
        p.error("--investigate-k, --max-consecutive-llm and --min-support must be >= 1")
    if args.pilot:
        args.max_env_steps = PILOT_STEPS
        if args.out_dir == "runs":
            args.out_dir = str(Path("runs") / "pilot")
    try:
        args.profile = resolve_profile(args.model_profile, args.llm, args.model, args.temperature,
                                       args.num_ctx, args.max_output_tokens, args.ollama_host)
    except (LLMSetupError, ValueError) as exc:
        p.error(str(exc))
    if args.llm and args.model_profile and args.llm != args.profile.backend:
        p.error(f"--llm {args.llm} conflicts with --model-profile {args.model_profile} "
                f"(backend {args.profile.backend})")
    args.llm = args.profile.backend
    args.temperature = args.profile.temperature
    if args.max_usd is not None and args.profile.usd(0, 0) is None:
        p.error(f"--max-usd needs known prices, but profile {args.profile.name!r} has none")
    if args.agent in LLM_AGENTS:
        prof = args.profile
        if prof.disabled:
            p.error(f"profile {prof.name!r} is disabled in configs/models.yaml "
                    "(edit the YAML to re-enable it)")
        if prof.usd(0, 0) is None and args.llm == "anthropic":
            p.error("an anthropic model with unknown price cannot be guarded; use a priced profile")
        if prof.is_paid:
            if not args.allow_paid:
                p.error(f"profile {prof.name!r} is paid (${prof.price_in_per_mtok}/"
                        f"${prof.price_out_per_mtok} per MTok); add --allow-paid to spend money")
            if args.max_usd is None:
                p.error("paid runs require --max-usd")
            if args.max_usd > PAID_MAX_USD_CEILING and not args.max_usd_override:
                p.error(f"--max-usd {args.max_usd} is above the ${PAID_MAX_USD_CEILING:.0f} per-run "
                        "ceiling; add --max-usd-override if you really mean it")
    try:
        args.seed_list = ([int(s) for s in args.seeds.split(",") if s.strip()]
                          if args.seeds else [args.seed])
    except ValueError:
        p.error("--seeds must be comma-separated integers")
    if args.run_dir and len(args.seed_list) != 1:
        p.error("--run-dir takes a single seed")
    return args


def make_agent(args, seed: int):
    """Returns (agent, backend, model); raises LLMSetupError."""
    if args.agent == "random":
        return RandomAgent(seed=seed), None, None
    if args.agent == "random-intent":
        return RandomIntentAgent(seed=seed), None, None
    if args.agent == "cheap":
        return CheapAgent(seed=seed), None, None
    intent = args.agent in ("react", "tkg")
    llm = make_llm_from_profile(args.profile, seed, intent_mock=intent)
    if args.agent == "tkg":
        trig = make_trigger(args.trigger, p=args.trigger_p, n=args.trigger_n, tau=args.tau,
                            min_support=args.min_support, seed=seed)
        agent = TKGAgent(llm, seed=seed, trigger=trig, warmup_steps=args.warmup_steps,
                         investigate_k=args.investigate_k, repro_path=args.repro_path,
                         history=args.history, min_support=args.min_support,
                         max_consecutive_llm=args.max_consecutive_llm,
                         warn=lambda m: print(m, file=sys.stderr))
        return agent, args.profile.backend, args.profile.model
    agent = ReActAgent(llm, history=args.history, actions_per_call=args.actions_per_call,
                       warn=lambda m: print(m, file=sys.stderr),
                       action_space="intent" if intent else "primitive",
                       intents_per_call=args.intents_per_call)
    return agent, args.profile.backend, args.profile.model


def paid_preflight(args):
    """For a paid profile: print the cost estimate and check the project ledger.
    Returns the Ledger to use, None for free runs; raises SpendGuardError."""
    if args.agent not in LLM_AGENTS or not args.profile.is_paid:
        return None
    est = spend.estimate_run(args.agent, args.profile, args.max_env_steps)
    print(f"[paid] {args.profile.name} ({args.profile.model}): estimate ${est['usd']:.4f} for "
          f"{args.max_env_steps} steps = {est['calls_per_step']:.2f} calls/step x "
          f"{est['tokens_in']:.0f} in + {est['tokens_out']:.0f} out tokens/call "
          f"[{est['source']}]; hard cap --max-usd ${args.max_usd:.2f}", flush=True)
    ledger = spend.Ledger(args.ledger or spend.LEDGER_PATH,
                          spend.load_budget(args.budget_file or spend.BUDGET_YAML))
    ledger.preflight(args.max_usd)
    print(f"[paid] budget OK: ${ledger.remaining():.4f} of ${ledger.budget.limit:.2f} usable remains "
          "before this run", flush=True)
    return ledger


def run_one(args, seed: int, ignore: list[int]):
    """Run a single seed. Returns (summary, run_dir), or (None, None) on a setup error."""
    try:
        ledger = paid_preflight(args)
    except spend.SpendGuardError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return None, None
    try:
        agent, backend, model = make_agent(args, seed)
    except LLMSetupError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return None, None

    cli_args = {k: v for k, v in vars(args).items() if k not in ("seed_list", "profile", "repro_config")}
    cli_args["seed"] = seed
    config = {"cli_args": cli_args, "backend": backend, "model": model,
              "command": " ".join(sys.argv),
              "model_profile": args.profile.to_dict() if args.agent in LLM_AGENTS else None}
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = make_run_dir(Path(args.out_dir), args.agent, args.task, seed)
    budgets = Budgets(args.max_env_steps, args.max_llm_calls, args.max_tokens, args.max_usd)

    def progress(stats, decision):
        if args.quiet:
            return
        rec = decision.llm_call
        flag = " PARSE_ERROR" if rec["parse_error"] else ""
        rep = f" | REPORT: {rec['parsed_bug_report']}" if rec["parsed_bug_report"] else ""
        if agent.action_space == "intent":
            did = " ; ".join(f"{r.intent} -> {r.n_steps} steps, {r.stop_reason}"
                             + (f" ({r.error})" if r.error else "") for r in decision.results or [])
        else:
            did = f"actions={[a for r in decision.results or [] for a in r.primitive_actions]}"
        usd = f" ${stats.usd:.4f}" if stats.usd else ""
        print(f"[seed {seed} | call {stats.llm_calls:>4} | step {stats.global_step:>5} | "
              f"ep {stats.episodes:>3}] {did} tok={stats.tokens}{usd} "
              f"bugs={len(stats.first_found)} {rec['latency_s']:.1f}s{flag}{rep}", flush=True)

    print(f"Run dir: {run_dir}")
    try:
        summary = run(agent, args.task, seed, budgets, run_dir, config,
                      ignore_bug_ids=ignore, progress=progress,
                      reader=ActionReader(args.max_macro_len), ledger=ledger,
                      repro=args.repro_config)
    except (LLMSetupError, ContextOverflowError, spend.SpendGuardError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return None, run_dir

    per = summary["parse_error_rate"]
    print(f"\n=== SUMMARY (seed {seed}) ===")
    if per is not None:
        print(f"PARSE ERROR RATE: {per:.1%} ({summary['parse_errors']}/{summary['llm_calls']} calls)")
    for k in ["agent", "action_space", "backend", "model", "stop_reason", "total_env_steps",
              "episodes", "decisions", "llm_calls", "llm_calls_per_env_step", "total_tokens",
              "usd_cost", "llm_latency_median_s", "intent_failure_rate", "n_unique_bugs",
              "unique_bug_ids", "deliveries_success", "agent_bug_reports", "reports_strict_match",
              "reports_lenient_match", "bugs_by_proposed_class", "wall_time_s"]:
        if k in summary:
            print(f"{k:>22}: {summary[k]}")
    for w in summary.get("taxonomy_warnings") or []:
        print(f"WARNING (draft taxonomy): {w}", file=sys.stderr)
    print(f"Logs: {run_dir}\n")
    return summary, run_dir


def main(argv=None):
    args = parse_args(argv)
    ignore = [int(x) for x in args.ignore_bug_ids.split(",") if x.strip()]
    summaries, run_dirs = [], []
    for seed in args.seed_list:
        summary, run_dir = run_one(args, seed, ignore)
        if summary is None:
            return 2
        summaries.append(summary)
        run_dirs.append(run_dir)

    if len(summaries) > 1:
        def agg(key):
            vals = [s[key] for s in summaries if s.get(key) is not None]
            if not vals:
                return "n/a"
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{statistics.mean(vals):.3g} ± {sd:.2g}"
        print(f"=== AGGREGATE over seeds {args.seed_list} (mean ± sd) ===")
        for k in ["n_unique_bugs", "deliveries_success", "total_tokens", "usd_cost",
                  "llm_calls", "agent_bug_reports", "parse_error_rate", "wall_time_s"]:
            print(f"{k:>20}: {agg(k)}")
        union = sorted({b for s in summaries for b in s["unique_bug_ids"]})
        print(f"{'union of bug IDs':>20}: {union}")

    if args.pilot:
        print("=== PILOT CHECK ===")
        return pilot_check.main([str(d) for d in run_dirs])
    return 0


if __name__ == "__main__":
    sys.exit(main())
