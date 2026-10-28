"""Agent-visible text uses the legend / STATIONS display names, never SAGE's internal names
or event types (e.g. "putdown knife succeeded" made the Haiku pilot file two false bug
reports), and never adds synonyms the legend does not use (e.g. "pan (stove)")."""
import json
import random
import re

import pytest

from react_baseline.action_reader import ActionReader, VisibleStep, valid_intents
from react_baseline.agents import ReActAgent
from react_baseline.env_adapter import (EnvAdapter, TILE_CHARS, TILE_DESCRIPTIONS, describe_event,
                                        display_name, list_tasks)
from react_baseline.llm import MockIntentLLM
from react_baseline.runner import Budgets, run

# internal tile names whose display name differs ("knife" -> "cutting board", ...)
RENAMED = {k: display_name(k) for k in TILE_CHARS if display_name(k) != k}
# SAGE event types / flags and synonyms that must never reach the agent
FORBIDDEN = [re.compile(p) for p in (r"putdown", r"pickup", r"pickup into plate", r"\bdump\b",
                                     r"succeeded", r"\bstove\b")]
# snake_case words are SAGE identifiers, except the Action reader's own stop reasons
READER_WORDS = {"episode_end", "unexpected_event", "length_cap"}


def _raw_hits(text: str) -> list[str]:
    hits = [p.pattern for p in FORBIDDEN if p.search(text)]
    hits += [w for w in re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", text) if w not in READER_WORDS]
    stripped = text
    for disp in sorted(set(RENAMED.values()), key=len, reverse=True):
        stripped = stripped.replace(disp, " ")
    hits += [raw for raw in RENAMED if re.search(rf"\b{re.escape(raw)}\b", stripped)]
    return hits


def test_display_name_mapping():
    assert display_name("knife") == "cutting board" and display_name("trash_can") == "trash can"
    assert display_name("delivery") == "delivery window" and display_name("ricecooker") == "rice cooker"
    assert display_name("pan") == "pan" and display_name("block") == "wall"
    assert display_name("tomato") == "tomato" and display_name("counter") == "counter"
    # every station's display name is the one the legend prints
    for raw in ("knife", "delivery", "pan", "sink", "trash_can", "ricecooker", "oven"):
        assert display_name(raw) == TILE_DESCRIPTIONS[raw]
    # the detector catches the old labels, and describe_event no longer produces them
    for old in ("putdown knife succeeded", "nothing happened (bumped trash_can)",
                "pickup into plate tomato failed", "dump plate+dough", "pan (stove): empty",
                "delivery REJECTED", "chop knife succeeded", "bumped block"):
        assert _raw_hits(old), old
    assert not _raw_hits("put tomato on the cutting board at (0,3); delivered the empty plate: "
                         "order REJECTED; pan: empty; threw away tomato in the trash can at (1,4)")
    info = {"item_log": [{"event_type": "None", "target": "trash_can", "details": {"others": {}}}]}
    assert describe_event(info) == "nothing happened (bumped trash can)"


def _record_texts(rec: dict) -> list[str]:
    """Agent-visible strings of a structured event record. ``type`` keeps SAGE's event names
    on purpose (stable structured field); text, target and tile/item names are checked."""
    out = [rec["text"], json.dumps(rec["target"])]
    for key in ("tile_before", "tile_after", "held_before", "held_after"):
        out.append(json.dumps(rec.get(key)).replace('"name"', '"n"').replace('"holding"', '"h"'))
    return out


def _held_lines_with_coords(obs: str) -> list[str]:
    return [line for line in obs.splitlines() if "held by you" in line and re.search(r"\(\d+,\d+\)", line)]


@pytest.mark.parametrize("task", list_tasks())
def test_no_internal_names_in_agent_visible_text(task):
    """Every agent-visible channel on every map: observation, history summary, structured
    event record, executor trace. Held items are listed without coordinates."""
    rng, reader = random.Random(0), ActionReader()
    adapter = EnvAdapter(task)
    texts, held_seen = [], 0
    try:
        texts.append(adapter.reset())
        state, steps = adapter.state, 0

        def step(a):
            r = adapter.step(a)
            texts.extend([r.obs_text, r.summary, *_record_texts(r.state.get("last_event"))])
            return VisibleStep(r.obs_text, r.summary, r.state, r.state["last_event"]["text"], r.done)

        while steps < 250:
            res = reader.execute(rng.choice(valid_intents(state)), state, step)
            steps += res.n_steps
            texts.append(res.trace)
            state = res.final.state
            if res.done:
                texts.append(adapter.reset())
                state = adapter.state
    finally:
        adapter.close()
    bad = [(t[:300], _raw_hits(t)) for t in texts if _raw_hits(t)]
    assert not bad, bad[:3]
    for t in texts:
        held_seen += "held by you" in t
        assert not _held_lines_with_coords(t), _held_lines_with_coords(t)
    assert all("Station letters do not show what the station holds; see STATIONS." in t
               for t in texts if t.startswith("MAP"))


@pytest.mark.parametrize("agent_kind", ["intent", "primitive"])
def test_no_internal_names_in_react_prompts(tmp_path, agent_kind):
    """The full prompts of both ReAct arms (history lines, markers, observation)."""
    llm = MockIntentLLM(seed=3) if agent_kind == "intent" else None
    if llm is None:
        from react_baseline.llm import MockLLM
        llm = MockLLM(seed=3)
    agent = ReActAgent(llm, history=10, action_space=agent_kind)
    for task in ("V2_task_07_tomato_steak", "V2_task_78_onion_fish_rice"):
        run(agent, task, 0, Budgets(max_env_steps=220), tmp_path / task, {"backend": "mock", "model": "mock"})
    prompts = [u for _, u in llm.calls]
    assert any(ReActAgent.RESET_MARKER in u for u in prompts)
    bad = [(u[-400:], _raw_hits(u)) for u in prompts if _raw_hits(u)]
    assert not bad, bad[:2]
    assert not any(_held_lines_with_coords(u) for u in prompts)
