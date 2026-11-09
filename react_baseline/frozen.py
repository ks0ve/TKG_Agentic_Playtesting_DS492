"""Frozen baseline configuration: what defines "baseline v2", and a check against the code.

    python -m react_baseline.frozen --check configs/frozen/baseline_v2_2.yaml
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
FROZEN_DEFAULT = REPO_ROOT / "configs" / "frozen" / "baseline_v2_2.yaml"


def prompt_texts() -> dict[str, str]:
    from . import prompts, repro
    from .agents import ReActAgent
    return {
        "reset_marker": ReActAgent.RESET_MARKER,
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
# tomato map: the thrown-away steak respawns on its start tile under/over the tomato, then is taken
RENDER_PROBE_STACK = ("interact(0,1)", "interact(1,0)", "interact(0,2)", "interact(0,1)", "interact(1,0)",
                      "interact(1,4)", "goto(1,1)", "interact(0,1)", "interact(1,0)", "interact(0,1)")


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
            if task == RENDER_PROBE_TASKS[0]:   # translator v2.2: two items on one counter tile
                adapter.reset()
                state = adapter.state
                for intent in RENDER_PROBE_STACK:
                    res = reader.execute(intent, state, step)
                    h.update(res.trace.encode())
                    state = res.final.state
        finally:
            adapter.close()
    return h.hexdigest()


IDENTITY_KEYS = ("prompts_sha256", "renderer_sha256", "action_reader_version", "translator_version")


def code_identity() -> dict:
    from .action_reader import ACTION_READER_VERSION
    from .env_adapter import TRANSLATOR_VERSION
    return {"prompts_sha256": prompt_hash(), "renderer_sha256": renderer_hash(),
            "action_reader_version": ACTION_READER_VERSION, "translator_version": TRANSLATOR_VERSION}


TKG_KEYS = ("memory_text_sha256", "reference_rules_sha256", "cheap_policy")
TKG_OPTIONAL_KEYS = ("routine_text_sha256",)     # checked only when the freeze has them (not tkg_v1)
TKG_SETTINGS = ("trigger", "trigger_n", "tau", "min_support", "warmup_steps", "investigate_k",
                "max_consecutive_llm", "repro_path")


def tkg_identity() -> dict:
    """What defines the TKG arm beyond the harness: the agent-visible MEMORY / recorded-path text
    (rendered samples + the source of the functions that build it), the game-manual rules file, and
    the cheap policy's parameters."""
    import inspect
    from memory.reference import RULES_YAML, Reference
    from .agents import ReActAgent
    from .cheap_policy import CheapPolicy
    from .tkg_agent import INVESTIGATE, ROUTINE_REASONS, expectation_line, memory_block, memory_block_for, path_block
    ref = Reference()
    t = {"actor": "cheap", "intent": "interact(0,2)", "event_text": "picked up tomato from the counter at (0,2)",
         "outcome": "pickup | - | hands:unchopped tomato | counter:empty"}
    lookups = [{"level": "reference", "hit": ref.lookup("interact | hands:nothing | counter:holding:unchopped tomato"),
                "dist": {}, "support": None},
               {"level": "ctx0", "dist": {t["outcome"]: 1.0}, "support": 4, "hit": None},
               {"level": "unseen", "dist": {}, "support": 0, "hit": None}]
    text = "\x00".join([INVESTIGATE, path_block([t])] + [memory_block(t, lk) for lk in lookups]
                        + [inspect.getsource(f) for f in (memory_block, expectation_line, path_block,
                                                          ReActAgent.observe_auto)])
    routine = "\x00".join([memory_block_for(t, lk, r) for lk in lookups for r in ROUTINE_REASONS]
                          + [inspect.getsource(memory_block_for)])
    params = inspect.signature(CheapPolicy.__init__).parameters
    return {"memory_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "reference_rules_sha256": hashlib.sha256(  # line endings normalised (git may check out CRLF)
                RULES_YAML.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "cheap_policy": {"epsilon": params["epsilon"].default, "n_goto": params["n_goto"].default},
            "routine_text_sha256": hashlib.sha256(routine.encode()).hexdigest()}


def load(path: Path | str = FROZEN_DEFAULT) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def check(path: Path | str = FROZEN_DEFAULT) -> list[str]:
    """-> list of mismatches between the frozen file and the current code (empty = OK)."""
    p = Path(path)
    if not p.exists():
        return [f"frozen config {p} not found"]
    frozen = load(p)
    if frozen.get("kind") == "tkg":
        code = tkg_identity()
        keys = TKG_KEYS + tuple(k for k in TKG_OPTIONAL_KEYS if k in frozen)
        return [f"{k}: frozen {frozen.get(k)!r} != code {code[k]!r}" for k in keys if frozen.get(k) != code[k]]
    code = code_identity()
    out = []
    for key in IDENTITY_KEYS:
        if frozen.get(key) != code[key]:
            out.append(f"{key}: frozen {frozen.get(key)!r} != code {code[key]!r}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Frozen baseline configuration check")
    ap.add_argument("--check", metavar="YAML", default=None)
    ap.add_argument("--print-identity", action="store_true")
    ap.add_argument("--print-tkg-identity", action="store_true")
    a = ap.parse_args(argv)
    if a.print_tkg_identity:
        print(yaml.safe_dump(tkg_identity(), sort_keys=False).strip())
        return 0
    if a.print_identity or not a.check:
        print(yaml.safe_dump(code_identity(), sort_keys=False).strip())
    if a.check:
        bad = check(a.check)
        print("FROZEN CONFIG MATCHES CODE" if not bad else "MISMATCH:\n  " + "\n  ".join(bad))
        return 0 if not bad else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
