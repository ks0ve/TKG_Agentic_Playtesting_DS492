"""Pack run directories into one zip for offline analysis elsewhere.

    python -m react_baseline.bundle <run_dir> [<run_dir> ...] -o pilots.zip

Per run: config.json, summary.json, steps.jsonl, llm_calls.jsonl, bug_reports.jsonl,
repro_attempts.jsonl, transitions.jsonl, episodes.jsonl (those that exist) and a fresh
pilot_check.txt. Once per bundle: spend/ledger.jsonl, the spend report, git HEAD +
status, and the grid --dry-run output. Works on Windows (zipfile, posix arcnames).
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import zipfile
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_FILES = ["config.json", "summary.json", "steps.jsonl", "llm_calls.jsonl", "bug_reports.jsonl",
             "repro_attempts.jsonl", "transitions.jsonl", "episodes.jsonl"]
DEFAULT_GRID = REPO_ROOT / "configs" / "grids" / "baseline_v2.yaml"


def _git(*args) -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                             timeout=20)
        return out.stdout if out.returncode == 0 else f"(git {' '.join(args)} failed: {out.stderr.strip()})"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(git unavailable: {exc})"


def _capture(fn) -> str:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            fn()
    except Exception as exc:  # the bundle should still be written
        buf.write(f"\n(failed: {type(exc).__name__}: {exc})\n")
    return buf.getvalue()


def build(run_dirs: list[Path | str], out: Path | str, grid_yaml: Path | str | None = DEFAULT_GRID) -> Path:
    from . import grid, pilot_check, spend
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = [f"bundle created {datetime.now().isoformat(timespec='seconds')}"]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for d in map(Path, run_dirs):
            if not d.is_dir():
                manifest.append(f"MISSING run dir: {d}")
                continue
            base = f"runs/{d.name}"
            for f in RUN_FILES:
                if (d / f).exists():
                    z.write(d / f, f"{base}/{f}")
            try:
                ok, lines = pilot_check.check(d)
                text = f"{'PILOT PASSED' if ok else 'PILOT FAILED'}\n" + "\n".join(lines) + "\n"
            except Exception as exc:
                text = f"pilot_check failed: {type(exc).__name__}: {exc}\n"
            z.writestr(f"{base}/pilot_check.txt", text)
            manifest.append(f"run {d.name}: " + text.splitlines()[0])
        ledger_path = spend.LEDGER_PATH
        if ledger_path.exists():
            z.write(ledger_path, "spend/ledger.jsonl")
        z.writestr("spend_report.txt", _capture(lambda: print(spend.Ledger().report())))
        z.writestr("git.txt", "HEAD " + _git("rev-parse", "HEAD") + "\n--- status --porcelain\n"
                   + _git("status", "--porcelain"))
        if grid_yaml and Path(grid_yaml).exists():
            z.writestr("grid_dry_run.txt", _capture(lambda: grid.main([str(grid_yaml), "--dry-run"])))
        z.writestr("MANIFEST.txt", "\n".join(manifest) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bundle run dirs for analysis")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--grid", default=str(DEFAULT_GRID))
    a = ap.parse_args(argv)
    path = build(a.run_dirs, a.out, a.grid)
    with zipfile.ZipFile(path) as z:
        n = len(z.namelist())
    print(f"wrote {path} ({n} files, {path.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
