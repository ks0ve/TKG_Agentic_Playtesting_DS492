"""Project spend guardrails for paid LLM runs.

    python -m react_baseline.spend                       # totals, remaining, breakdowns
    python -m react_baseline.spend --reconcile <run_dir> # finalize a stale reservation

``configs/budget.yaml`` sets ``project_cap_usd`` and ``safety_margin_usd``. Paid runs may
only start while  finalized spend + open reservations + this run's --max-usd
<= cap - margin. ``spend/ledger.jsonl`` (tracked in git) is append-only:
  {"type": "reservation", "run_dir", "profile", "model", "max_usd", "ts", "status": "running"}
  {"type": "final", "run_dir", "usd_cost", "input_tokens", "output_tokens", "status", "stop_reason", "ts"}
A reservation without a final record (crash, killed process) counts at its full
max_usd until reconciled from the run's summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BUDGET_YAML = REPO_ROOT / "configs" / "budget.yaml"
LEDGER_PATH = REPO_ROOT / "spend" / "ledger.jsonl"
PILOT_ROOT = REPO_ROOT / "runs" / "pilot"

# conservative per-call defaults when no pilot measurement exists (labelled as such)
DEFAULT_CALLS_PER_STEP = 1.0
DEFAULT_TOKENS_IN = 3000
DEFAULT_TOKENS_OUT = 200
DEFAULT_LATENCY_S = 30.0       # local models on the laptop
DEFAULT_API_LATENCY_S = 3.0    # hosted API


class SpendGuardError(RuntimeError):
    """A paid run would breach the project budget (or the ledger is unusable)."""


@dataclass
class Budget:
    project_cap_usd: float
    safety_margin_usd: float

    @property
    def limit(self) -> float:
        return self.project_cap_usd - self.safety_margin_usd


def load_budget(path: Path | str = BUDGET_YAML) -> Budget:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Budget(float(data["project_cap_usd"]), float(data.get("safety_margin_usd", 0.0)))


def _key(run_dir) -> str:
    """Stable ledger key: repo-relative posix path when possible."""
    p = Path(run_dir).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path | str = LEDGER_PATH, budget: Budget | None = None):
        self.path = Path(path)
        self.budget = budget or load_budget()

    # -- reading -------------------------------------------------------------
    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SpendGuardError(f"{self.path}:{i} is not valid JSON; fix the ledger "
                                          "before any paid run") from exc
        return out

    def state(self) -> tuple[dict, dict]:
        """-> (reservations by run_dir, finals by run_dir)."""
        res, fin = {}, {}
        for r in self.records():
            (res if r.get("type") == "reservation" else fin)[r["run_dir"]] = r
        return res, fin

    def spent(self) -> float:
        _, fin = self.state()
        return sum(float(r.get("usd_cost") or 0.0) for r in fin.values())

    def open_reservations(self) -> list[dict]:
        res, fin = self.state()
        return [r for k, r in res.items() if k not in fin]

    def committed(self) -> float:
        return self.spent() + sum(float(r["max_usd"]) for r in self.open_reservations())

    def remaining(self) -> float:
        return self.budget.limit - self.committed()

    # -- guarding / writing --------------------------------------------------
    def preflight(self, max_usd: float):
        total = self.committed() + float(max_usd)
        if total > self.budget.limit + 1e-9:
            raise SpendGuardError(
                f"Refusing to start a paid run: spent ${self.spent():.4f} + open reservations "
                f"${self.committed() - self.spent():.4f} + this run's --max-usd ${max_usd:.2f} = "
                f"${total:.4f} > cap ${self.budget.project_cap_usd:.2f} - margin "
                f"${self.budget.safety_margin_usd:.2f} = ${self.budget.limit:.2f}. "
                "Check `python -m react_baseline.spend` (reconcile stale runs) or lower --max-usd.")

    def _append(self, rec: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def reserve(self, run_dir, profile: str, model: str, max_usd: float):
        self.preflight(max_usd)
        self._append({"type": "reservation", "run_dir": _key(run_dir), "profile": profile,
                      "model": model, "max_usd": float(max_usd), "ts": _now(), "status": "running"})

    def finalize(self, run_dir, usd_cost, input_tokens: int, output_tokens: int, status: str,
                 stop_reason: str | None = None):
        self._append({"type": "final", "run_dir": _key(run_dir),
                      "usd_cost": None if usd_cost is None else float(usd_cost),
                      "input_tokens": int(input_tokens), "output_tokens": int(output_tokens),
                      "status": status, "stop_reason": stop_reason, "ts": _now()})

    def reconcile(self, run_dir) -> dict:
        key = _key(run_dir)
        res, fin = self.state()
        if key not in res:
            raise SpendGuardError(f"No reservation for {key} in {self.path}")
        if key in fin:
            raise SpendGuardError(f"{key} is already finalized")
        summary_path = Path(run_dir) / "summary.json"
        if not summary_path.exists():
            raise SpendGuardError(f"{summary_path} not found; cannot reconcile. If the run never "
                                  "made a call, finalize it by hand with usd_cost 0.")
        s = json.loads(summary_path.read_text())
        self.finalize(run_dir, s.get("usd_cost"), s.get("input_tokens", 0), s.get("output_tokens", 0),
                      "reconciled", s.get("stop_reason"))
        return s

    # -- reporting -----------------------------------------------------------
    def report(self) -> str:
        res, fin = self.state()
        lines = [f"Project cap ${self.budget.project_cap_usd:.2f}, safety margin "
                 f"${self.budget.safety_margin_usd:.2f} -> usable ${self.budget.limit:.2f}",
                 f"Spent (finalized):      ${self.spent():.4f}",
                 f"Open reservations:      ${self.committed() - self.spent():.4f}",
                 f"Remaining for new runs: ${self.remaining():.4f}", ""]
        by_prof = defaultdict(float)
        for k, r in fin.items():
            by_prof[res.get(k, {}).get("profile", "?")] += float(r.get("usd_cost") or 0.0)
        lines.append("Per profile: " + (", ".join(f"{p} ${v:.4f}" for p, v in sorted(by_prof.items()))
                                        or "(none)"))
        lines.append("Per run:")
        for k, r in res.items():
            f = fin.get(k)
            if f:
                lines.append(f"  {k}  {r['profile']}  ${float(f.get('usd_cost') or 0):.4f} "
                             f"(cap ${r['max_usd']:.2f})  {f['status']}")
            else:
                lines.append(f"  {k}  {r['profile']}  OPEN (counts ${r['max_usd']:.2f})  "
                             f"started {r['ts']}")
        if not res:
            lines.append("  (no paid runs yet)")
        stale = self.open_reservations()
        if stale:
            lines.append("")
            lines.append("Open reservations (running, or crashed: reconcile with "
                         "`python -m react_baseline.spend --reconcile <run_dir>`):")
            lines += [f"  {r['run_dir']}  max ${r['max_usd']:.2f}  since {r['ts']}" for r in stale]
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Estimates (pilot measurements when available, otherwise labelled defaults)
def pilot_measurement(agent: str, model: str | None, pilot_root: Path | str = PILOT_ROOT) -> dict | None:
    """Per-call numbers from the most recent pilot run of ``agent`` with ``model``."""
    root = Path(pilot_root)
    if not root.exists():
        return None
    for d in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        sp = d / "summary.json"
        if not sp.exists():
            continue
        try:
            s = json.loads(sp.read_text())
        except json.JSONDecodeError:
            continue
        if s.get("agent") != agent or s.get("model") != model:
            continue
        calls, steps = s.get("llm_calls") or 0, s.get("total_env_steps") or 0
        if model is not None and (not calls or not steps):
            continue
        return {"source": f"pilot {d.name}",
                "calls_per_step": calls / steps if steps else 0.0,
                "tokens_in": (s.get("input_tokens") or 0) / calls if calls else 0.0,
                "tokens_out": (s.get("output_tokens") or 0) / calls if calls else 0.0,
                "latency_s": s.get("llm_latency_median_s") or 0.0,
                "wall_per_step_s": (s.get("wall_time_s") or 0.0) / steps if steps else 0.0}
    return None


def estimate_run(agent: str, profile, steps: int, pilot_root: Path | str = PILOT_ROOT) -> dict:
    """-> {usd, wall_s, calls, source, calls_per_step, tokens_in, tokens_out, latency_s}."""
    model = getattr(profile, "model", None) if profile is not None else None
    m = pilot_measurement(agent, model, pilot_root)
    if m is None:
        if profile is None:  # no-LLM agents: measured ~1.5 ms per step offline
            m = {"source": "default (no LLM)", "calls_per_step": 0.0, "tokens_in": 0, "tokens_out": 0,
                 "latency_s": 0.0, "wall_per_step_s": 0.002}
        else:
            lat = DEFAULT_API_LATENCY_S if profile.backend == "anthropic" else DEFAULT_LATENCY_S
            m = {"source": "conservative default (no pilot found)",
                 "calls_per_step": DEFAULT_CALLS_PER_STEP, "tokens_in": DEFAULT_TOKENS_IN,
                 "tokens_out": DEFAULT_TOKENS_OUT, "latency_s": lat,
                 "wall_per_step_s": DEFAULT_CALLS_PER_STEP * lat}
    calls = steps * m["calls_per_step"]
    usd = 0.0
    if profile is not None:
        per_call = profile.usd(m["tokens_in"], m["tokens_out"])
        usd = None if per_call is None else calls * per_call
    wall = steps * m["wall_per_step_s"] if m["wall_per_step_s"] else calls * m["latency_s"]
    return {**m, "calls": calls, "usd": usd, "wall_s": wall}


def main(argv=None):
    p = argparse.ArgumentParser(description="Project spend ledger for paid LLM runs")
    p.add_argument("--reconcile", metavar="RUN_DIR", default=None,
                   help="finalize a stale reservation from RUN_DIR/summary.json")
    p.add_argument("--ledger", default=str(LEDGER_PATH))
    p.add_argument("--budget", default=str(BUDGET_YAML))
    args = p.parse_args(argv)
    ledger = Ledger(args.ledger, load_budget(args.budget))
    try:
        if args.reconcile:
            s = ledger.reconcile(args.reconcile)
            print(f"Reconciled {args.reconcile}: usd_cost {s.get('usd_cost')}")
        print(ledger.report())
    except SpendGuardError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
