"""CLI: python -m react_baseline.run --agent {react,random} --task <id> ..."""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from .agents import ContextOverflowError, RandomAgent, ReActAgent
from .env_adapter import list_tasks
from .llm import BACKENDS, MODELS_YAML, LLMSetupError, make_llm_from_profile, resolve_profile
from .runner import Budgets, make_run_dir, run


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ReAct / random bug-hunting baseline on Overcooked Plus")
    p.add_argument("--agent", choices=["react", "random"], required=True)
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
    p.add_argument("--actions-per-call", type=int, default=1)
    p.add_argument("--num-ctx", type=int, default=None,
                   help="context window (default: the profile's; for Ollama also num_ctx)")
    p.add_argument("--ollama-host", default=None, help="override the profile's base_url")
    p.add_argument("--ignore-bug-ids", default="5",
                   help="comma-separated oracle bug IDs to drop from logs/metrics "
                        "(default 5: fires on every completed chop on single-knife maps; see README)")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    if args.task not in list_tasks():
        p.error(f"unknown task {args.task!r}; see external/SAGE/scenarios/overcooked_plus/tasks.json")
    if args.actions_per_call < 1:
        p.error("--actions-per-call must be >= 1")
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
    try:
        args.seed_list = ([int(s) for s in args.seeds.split(",") if s.strip()]
                          if args.seeds else [args.seed])
    except ValueError:
        p.error("--seeds must be comma-separated integers")
    return args


def run_one(args, seed: int, ignore: list[int]):
    """Run a single seed. Returns (summary, run_dir), or (None, None) on a setup error."""
    if args.agent == "random":
        agent = RandomAgent(seed=seed)
        backend, model = None, None
    else:
        backend, model = args.profile.backend, args.profile.model
        try:
            llm = make_llm_from_profile(args.profile, seed)
        except LLMSetupError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return None, None
        agent = ReActAgent(llm, history=args.history, actions_per_call=args.actions_per_call,
                           warn=lambda m: print(m, file=sys.stderr))

    cli_args = {k: v for k, v in vars(args).items() if k not in ("seed_list", "profile")}
    cli_args["seed"] = seed
    config = {"cli_args": cli_args, "backend": backend, "model": model,
              "command": " ".join(sys.argv),
              "model_profile": args.profile.to_dict() if args.agent != "random" else None}
    run_dir = make_run_dir(Path(args.out_dir), args.agent, args.task, seed)
    budgets = Budgets(args.max_env_steps, args.max_llm_calls, args.max_tokens, args.max_usd)

    def progress(stats, decision):
        if args.quiet:
            return
        rec = decision.llm_call
        flag = " PARSE_ERROR" if rec["parse_error"] else ""
        rep = f" | REPORT: {rec['parsed_bug_report']}" if rec["parsed_bug_report"] else ""
        print(f"[seed {seed} | call {stats.llm_calls:>4} | step {stats.global_step:>5} | "
              f"ep {stats.episodes:>3}] actions={decision.actions} tok={stats.tokens} "
              f"bugs={len(stats.first_found)} {rec['latency_s']:.1f}s{flag}{rep}", flush=True)

    print(f"Run dir: {run_dir}")
    try:
        summary = run(agent, args.task, seed, budgets, run_dir, config,
                      ignore_bug_ids=ignore, progress=progress)
    except (LLMSetupError, ContextOverflowError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return None, run_dir

    per = summary["parse_error_rate"]
    print(f"\n=== SUMMARY (seed {seed}) ===")
    if per is not None:
        print(f"PARSE ERROR RATE: {per:.1%} ({summary['parse_errors']}/{summary['llm_calls']} calls)")
    for k in ["agent", "backend", "model", "stop_reason", "total_env_steps", "episodes",
              "llm_calls", "total_tokens", "usd_cost", "llm_latency_median_s", "n_unique_bugs", "unique_bug_ids",
              "deliveries_success", "agent_bug_reports", "wall_time_s"]:
        print(f"{k:>20}: {summary[k]}")
    print(f"Logs: {run_dir}\n")
    return summary, run_dir


def main(argv=None):
    args = parse_args(argv)
    ignore = [int(x) for x in args.ignore_bug_ids.split(",") if x.strip()]
    summaries = []
    for seed in args.seed_list:
        summary, _ = run_one(args, seed, ignore)
        if summary is None:
            return 2
        summaries.append(summary)

    if len(summaries) > 1:
        def agg(key):
            vals = [s[key] for s in summaries if s[key] is not None]
            if not vals:
                return "n/a"
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{statistics.mean(vals):.3g} ± {sd:.2g}"
        print(f"=== AGGREGATE over seeds {args.seed_list} (mean ± sd) ===")
        for k in ["n_unique_bugs", "deliveries_success", "total_tokens",
                  "agent_bug_reports", "parse_error_rate", "wall_time_s"]:
            print(f"{k:>20}: {agg(k)}")
        union = sorted({b for s in summaries for b in s["unique_bug_ids"]})
        print(f"{'union of bug IDs':>20}: {union}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
