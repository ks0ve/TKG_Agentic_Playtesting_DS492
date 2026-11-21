"""Post-hoc rescoring with the order-independent normal-play sets (added fields only).

    python -m react_baseline.rescore runs/grids/baseline_v2          # every run in a grid
    python -m react_baseline.rescore runs/grids/baseline_v2/<run>    # one run

For each run this writes ``summary_rescored.json`` next to ``summary.json``: the original
summary plus the ``*_adj`` fields below and a ``rescore`` block saying which normal-play file
was used. ``summary.json`` is never modified. Adjusted = the task's normal-play IDs
(``configs/normal_play_bugs.yaml``) are removed; the oracle itself and ``ignore_bug_ids`` are
unchanged, and nothing is re-run.

  unique_bugs_excluding_normal_play   unique oracle IDs minus the normal-play set
  reports_strict_match_adj            reports whose strict evidence has an ID outside the set
  unique_bugs_reported_strict_adj     unique such IDs over all reports
  unique_bugs_discovered_adj          unique adjusted IDs both triggered and reported (strict)
  source_bug_ids_strict_adj           per repro attempt: the source report's strict IDs minus the set
  oracle_reproduced_adj               per repro attempt: one of those fired again (None if none left)
  repro_success_rate_adj              mean oracle_reproduced_adj over attempts with a non-empty target
  bugs_by_proposed_class_adj          draft-taxonomy classes of the adjusted unique IDs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from .solutions import NORMAL_PLAY_YAML, load_normal_play
from .taxonomy import taxonomy_counts


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def adjusted_fields(summary: dict, reports: list[dict], attempts: list[dict], normal) -> dict:
    """The *_adj fields for one run. ``normal`` is the task's normal-play set (None: unknown)."""
    if normal is None:
        return {"unique_bugs_excluding_normal_play": None, "normal_play_bug_ids": None,
                "reports_strict_match_adj": None, "unique_bugs_reported_strict_adj": None,
                "unique_bug_ids_reported_strict_adj": None, "unique_bugs_discovered_adj": None,
                "repro_attempts_adj": None,
                "repro_success_rate_adj": None, "bugs_by_proposed_class_adj": None}
    normal = set(normal)
    found = set(summary.get("unique_bug_ids") or []) - normal
    strict_adj = [sorted(set(r.get("bug_ids_strict") or []) - normal) for r in reports]
    attempts_adj = []
    for a in attempts:
        target = sorted(set(a.get("source_bug_ids_strict") or []) - normal)
        again = sorted(set(a.get("reproduced_bug_ids") or []) - normal)
        attempts_adj.append({"attempt_id": a.get("attempt_id"), "source_bug_ids_strict_adj": target,
                             "oracle_reproduced_adj": bool(again) if target else None,
                             "reproduced_bug_ids_adj": again})
    scored = [a for a in attempts_adj if a["oracle_reproduced_adj"] is not None]
    return {
        "unique_bugs_excluding_normal_play": len(found),
        "normal_play_bug_ids": sorted(normal),
        "unique_bug_ids_excluding_normal_play": sorted(found),
        "reports_strict_match_adj": sum(1 for s in strict_adj if s),
        "unique_bugs_reported_strict_adj": len({b for s in strict_adj for b in s}),
        "unique_bug_ids_reported_strict_adj": sorted({b for s in strict_adj for b in s}),
        "unique_bugs_discovered_adj": len(found & {b for s in strict_adj for b in s}),
        "repro_attempts_adj": attempts_adj,
        "repro_attempts_scored_adj": len(scored),
        "repro_success_rate_adj": (sum(a["oracle_reproduced_adj"] for a in scored) / len(scored)) if scored else None,
        "bugs_by_proposed_class_adj": taxonomy_counts(found)["bugs_by_proposed_class"],
    }


def rescore_run(run_dir: Path | str, normal_play: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    normal_play = load_normal_play() if normal_play is None else normal_play
    out = {**summary,
           **adjusted_fields(summary, _jsonl(run_dir / "bug_reports.jsonl"),
                             _jsonl(run_dir / "repro_attempts.jsonl"), normal_play.get(summary["task"])),
           "rescore": {"normal_play_file": str(NORMAL_PLAY_YAML.relative_to(NORMAL_PLAY_YAML.parents[1])),
                       "normal_play_sha256": hashlib.sha256(NORMAL_PLAY_YAML.read_bytes()).hexdigest()
                       if NORMAL_PLAY_YAML.exists() else None,
                       "rescored_at": datetime.now().isoformat(timespec="seconds")}}
    (run_dir / "summary_rescored.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_dirs(path: Path) -> list[Path]:
    if (path / "summary.json").exists():
        return [path]
    return sorted(d for d in path.iterdir()
                  if d.is_dir() and (d / "summary.json").exists() and "_stale" not in d.name)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def table(rows: list[dict]) -> str:
    """Arm x task means over seeds."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        arm = r["agent"] + (f"/{r['model_profile']}" if r.get("model_profile") else "")
        groups.setdefault((arm, r["task"]), []).append(r)
    head = ("| arm | task | seeds | bugs raw | bugs adj | deliveries | LLM calls | $ | "
            "reports strict adj | repro success adj |")
    lines = [head, "|" + "---|" * 10]

    def f(x, nd=1):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    for (arm, task), rs in sorted(groups.items()):
        lines.append("| " + " | ".join([
            arm, task.replace("V2_task_", ""), str(len(rs)),
            f(_mean([r["n_unique_bugs"] for r in rs])),
            f(_mean([r["unique_bugs_excluding_normal_play"] for r in rs])),
            f(_mean([r["deliveries_success"] for r in rs])),
            f(_mean([r["llm_calls"] for r in rs]), 0),
            f(_mean([r.get("usd_cost") for r in rs]), 3),
            f(_mean([r["reports_strict_match_adj"] for r in rs])),
            f(_mean([r["repro_success_rate_adj"] for r in rs]), 2),
        ]) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", help="a run directory or a grid directory of runs")
    args = p.parse_args(argv)
    path = Path(args.path)
    dirs = run_dirs(path)
    if not dirs:
        print(f"no runs with a summary.json under {path}", file=sys.stderr)
        return 2
    normal = load_normal_play()
    rows = []
    for d in dirs:
        r = rescore_run(d, normal)
        cfg_path = d / "config.json"
        if cfg_path.exists():
            prof = (json.loads(cfg_path.read_text(encoding="utf-8")).get("model_profile") or {}).get("name")
            r["model_profile"] = prof
        rows.append(r)
    print(f"wrote summary_rescored.json in {len(dirs)} run dir(s) (summary.json untouched)\n")
    print("Means over seeds. bugs adj = unique oracle IDs outside the task's normal-play set; "
          "repro success adj = over attempts whose source report had such an ID.\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
