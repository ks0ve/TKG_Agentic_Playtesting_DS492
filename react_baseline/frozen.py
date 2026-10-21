"""Frozen baseline configuration: what defines "baseline v2", and a check against the code.

    python -m react_baseline.frozen --check configs/frozen/baseline_v2.yaml
    python -m react_baseline.frozen --print-identity

The grid driver refuses to run when the code's prompt hash or Action-reader version
differs from the frozen file (unless --allow-unfrozen), so full runs cannot silently
mix baseline versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_DEFAULT = REPO_ROOT / "configs" / "frozen" / "baseline_v2.yaml"


def prompt_texts() -> dict[str, str]:
    from . import prompts, repro
    return {
        "system_intent_1": prompts.system_prompt(1, "intent"),
        "system_primitive_1": prompts.system_prompt(1, "primitive"),
        "turn_template": prompts.TURN_TEMPLATE,
        "format_reminder": prompts.FORMAT_REMINDER,
        "action_spec_intent_1": prompts.action_spec(1, "intent"),
        "action_spec_primitive_1": prompts.action_spec(1, "primitive"),
        "repro_task_template": repro.TASK_TEMPLATE,
        "repro_markers": repro.START_MARKER + "|" + repro.END_MARKER,
    }


def prompt_hash() -> str:
    blob = json.dumps(prompt_texts(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def code_identity() -> dict:
    from .action_reader import ACTION_READER_VERSION
    return {"prompts_sha256": prompt_hash(), "action_reader_version": ACTION_READER_VERSION}


def load(path: Path | str = FROZEN_DEFAULT) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def check(path: Path | str = FROZEN_DEFAULT) -> list[str]:
    """-> list of mismatches between the frozen file and the current code (empty = OK)."""
    p = Path(path)
    if not p.exists():
        return [f"frozen config {p} not found"]
    frozen, code = load(p), code_identity()
    out = []
    for key in ("prompts_sha256", "action_reader_version"):
        if frozen.get(key) != code[key]:
            out.append(f"{key}: frozen {frozen.get(key)!r} != code {code[key]!r}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Frozen baseline configuration check")
    ap.add_argument("--check", metavar="YAML", default=None)
    ap.add_argument("--print-identity", action="store_true")
    a = ap.parse_args(argv)
    if a.print_identity or not a.check:
        print(yaml.safe_dump(code_identity(), sort_keys=False).strip())
    if a.check:
        bad = check(a.check)
        print("FROZEN CONFIG MATCHES CODE" if not bad else "MISMATCH:\n  " + "\n  ".join(bad))
        return 0 if not bad else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
