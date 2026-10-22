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


RENDER_PROBE_TASKS = ("V2_task_01_tomato", "V2_task_07_tomato_steak", "V2_task_11_tomato_lettuce_onion")


def renderer_hash() -> str:
    """Hash of every agent-visible text (observation, history summary, event, executor trace)
    produced by replaying the scripted solutions of the probe tasks, plus the display names.
    Changes whenever the renderer or event wording changes."""
    from .action_reader import ActionReader, VisibleStep
    from .env_adapter import EnvAdapter, display_name, TILE_DESCRIPTIONS
    from .solutions import load_solution

    h = hashlib.sha256()
    names = {k: display_name(k) for k in sorted(TILE_DESCRIPTIONS)}
    h.update(json.dumps(names, sort_keys=True).encode())
    for task in RENDER_PROBE_TASKS:
        adapter = EnvAdapter(task)
        try:
            h.update(adapter.reset().encode())
            state, reader = adapter.state, ActionReader()

            def step(a):
                r = adapter.step(a)
                h.update((r.obs_text + "\x00" + r.summary).encode())
                return VisibleStep(r.obs_text, r.summary, r.state, r.state["last_event"]["text"], r.done)

            for intent in load_solution(task):
                res = reader.execute(intent, state, step)
                h.update(res.trace.encode())
                state = res.final.state
                if res.done:
                    break
        finally:
            adapter.close()
    return h.hexdigest()


def code_identity() -> dict:
    from .action_reader import ACTION_READER_VERSION
    return {"prompts_sha256": prompt_hash(), "renderer_sha256": renderer_hash(),
            "action_reader_version": ACTION_READER_VERSION}


def load(path: Path | str = FROZEN_DEFAULT) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def check(path: Path | str = FROZEN_DEFAULT) -> list[str]:
    """-> list of mismatches between the frozen file and the current code (empty = OK)."""
    p = Path(path)
    if not p.exists():
        return [f"frozen config {p} not found"]
    frozen, code = load(p), code_identity()
    out = []
    for key in ("prompts_sha256", "renderer_sha256", "action_reader_version"):
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
