"""Lightweight, resumable grid driver.

    python -m react_baseline.grid configs/grids/baseline_v2_1.yaml --dry-run
    python -m react_baseline.grid configs/grids/baseline_v2_1.yaml [--only react/llama3.1-8b] [--allow-paid]

Runs agent x profile x task x seed combinations sequentially through the normal CLI
(so every guardrail applies). A combination is done when its run dir holds a
summary.json with stop_reason "max_env_steps" and the same config hash; otherwise it
is (re-)run, and an incomplete old dir is kept under "<dir>_stale<timestamp>".
Refuses to run when the code no longer matches the grid's frozen config, unless
--allow-unfrozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from . import frozen, spend
from .llm import resolve_profile

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Combo:
    agent: str
    profile: str | None
    task: str
    seed: int
    argv: list[str]
    run_dir: Path
    config_hash: str
    paid: bool

    @property
    def label(self) -> str:
        return f"{self.agent}/{self.profile or '-'}/{self.task}/seed{self.seed}"


def load_grid(path) -> dict:
    """Load a grid YAML. ``base: <yaml>`` inherits every setting from another grid (only keys
    given here override it), so a sub-grid produces the same run dirs and config hashes."""
    g = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if g.get("base"):
        base_path = Path(g["base"])
        if not base_path.is_absolute():
            base_path = REPO_ROOT / base_path
        merged = load_grid(base_path)
        merged.update({k: v for k, v in g.items() if k != "base"})
        g = merged
    return g


def _matches(only: str | None, agent: str, pname) -> bool:
    if not only:
        return True
    keys = {agent, f"{agent}/{pname}"}
    return any(o.strip() in keys for o in only.split(","))


def _base_argv(g: dict, agent: str, prof, task: str, seed: int) -> list[str]:
    argv = ["--agent", agent, "--task", task, "--seed", str(seed),
            "--max-env-steps", str(g["max_env_steps"]), "--history", str(g.get("history", 10)),
            "--intents-per-call", str(g.get("intents_per_call", 1)),
            "--max-macro-len", str(g.get("max_macro_len", 40))]
    if prof is not None:
        argv += ["--model-profile", prof.name]
        if prof.backend != "anthropic" or prof.options.get("send_temperature", True):
            argv += ["--temperature", str(g.get("temperature", 0.7))]
        if prof.is_paid:
            argv += ["--allow-paid", "--max-usd", f"{float(g['max_usd_per_paid_run']):.2f}"]
    r = g.get("repro") or {}
    if r.get("enabled"):
        argv += ["--repro", "--repro-steps", str(r.get("steps", 100)),
                 "--max-repro-attempts", str(r.get("max_attempts", 5)),
                 "--max-repro-fraction", str(r.get("max_fraction", 0.3)),
                 "--repro-dedup-jaccard", str(r.get("dedup_jaccard", 0.8))]
    return argv


def combos(g: dict, only: str | None = None) -> list[Combo]:
    identity = frozen.code_identity()
    out_root = REPO_ROOT / g.get("out_dir", f"runs/grids/{g['name']}")
    res = []
    for row in g["rows"]:
        agent, pname = row["agent"], row.get("profile")
        if not _matches(only, agent, pname):
            continue
        prof = resolve_profile(pname) if pname else None
        for task in row.get("tasks", g["tasks"]):
            for seed in row["seeds"]:
                argv = _base_argv(g, agent, prof, task, seed)
                h = hashlib.sha256(json.dumps({"argv": argv, "code": identity,
                                               "model": prof.model if prof else None},
                                              sort_keys=True).encode()).hexdigest()[:16]
                run_dir = out_root / f"{agent}_{pname or 'none'}_{task}_seed{seed}"
                res.append(Combo(agent, pname, task, seed, argv, run_dir, h,
                                 bool(prof and prof.is_paid)))
    return res


def status(c: Combo) -> str:
    if not c.run_dir.exists():
        return "pending"
    try:
        s = json.loads((c.run_dir / "summary.json").read_text())
        cfg = json.loads((c.run_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return "stale"
    same = cfg.get("cli_args", {}).get("grid_config_hash") == c.config_hash
    return "done" if same and s.get("stop_reason") == "max_env_steps" else "stale"


def _fmt_h(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def dry_run(g: dict, cs: list[Combo]) -> str:
    lines, tot_wall, tot_usd = [], 0.0, 0.0
    lines.append(f"{'status':8} {'est. wall':>9} {'est. $':>8}  combination  [estimate source]")
    for c in cs:
        st = status(c)
        prof = resolve_profile(c.profile) if c.profile else None
        est = spend.estimate_run(c.agent, prof, int(g["max_env_steps"]))
        usd = est["usd"] or 0.0
        if st != "done":
            tot_wall += est["wall_s"]
            tot_usd += usd
        lines.append(f"{st:8} {_fmt_h(est['wall_s']):>9} {usd:8.3f}  {c.label}  [{est['source']}]"
                     + ("  PAID" if c.paid else ""))
    n_todo = sum(status(c) != "done" for c in cs)
    lines.append(f"\n{len(cs)} combinations, {n_todo} to run: est. {_fmt_h(tot_wall)} wall, "
                 f"${tot_usd:.2f} (paid runs capped at ${float(g['max_usd_per_paid_run']):.2f} each)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Resumable experiment grid")
    ap.add_argument("grid")
    ap.add_argument("--only", default=None,
                    help="comma-separated agents or agent/profile pairs, e.g. "
                         "react/haiku-5.5,react-raw/haiku-5.5")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-paid", action="store_true", help="run paid combinations (guarded)")
    ap.add_argument("--allow-unfrozen", action="store_true",
                    help="run even if the code no longer matches the frozen config")
    a = ap.parse_args(argv)
    g = load_grid(a.grid)
    cs = combos(g, a.only)
    frozen_path = REPO_ROOT / g["frozen"] if g.get("frozen") else None
    mismatch = frozen.check(frozen_path) if frozen_path else ["grid has no 'frozen' entry"]
    if a.dry_run:
        print(f"grid {g['name']}: frozen check: {'OK' if not mismatch else 'MISMATCH ' + '; '.join(mismatch)}")
        print(dry_run(g, cs))
        return 0
    if mismatch and not a.allow_unfrozen:
        print("ERROR: code does not match the frozen config:\n  " + "\n  ".join(mismatch)
              + "\nRefusing to run (use --allow-unfrozen to override).", file=sys.stderr)
        return 2
    from . import run as run_cli
    for c in cs:
        st = status(c)
        if st == "done":
            print(f"[grid] done    {c.label}")
            continue
        if c.paid and not a.allow_paid:
            print(f"[grid] skip    {c.label} (paid; add --allow-paid)")
            continue
        if c.run_dir.exists():
            c.run_dir.rename(c.run_dir.with_name(c.run_dir.name + "_stale"
                                                 + datetime.now().strftime("%Y%m%d-%H%M%S")))
        print(f"[grid] run     {c.label}", flush=True)
        rc = run_cli.main(c.argv + ["--run-dir", str(c.run_dir), "--grid-config-hash",
                                    c.config_hash, "--quiet"])
        summ = c.run_dir / "summary.json"
        stop = json.loads(summ.read_text()).get("stop_reason") if summ.exists() else None
        if stop == "interrupted":
            print("[grid] interrupted; stopping (rerun the same command to resume)")
            return 130
        if rc != 0 and c.paid and not summ.exists():
            print("[grid] paid run refused (budget/guardrail); stopping the grid", file=sys.stderr)
            return rc
        if rc != 0:
            print(f"[grid] {c.label} failed (rc {rc}); continuing", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
