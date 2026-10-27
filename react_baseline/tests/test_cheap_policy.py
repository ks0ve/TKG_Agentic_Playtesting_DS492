"""Cheap policy: prefers less-tried situations (TKG counts), emits only valid intents."""
import json
import random

from memory.abstraction import predicted_ctx0
from memory.tkg import Expectation, TKG
from react_baseline.action_reader import valid_intents
from react_baseline.cheap_policy import CheapAgent, CheapPolicy
from react_baseline.env_adapter import EnvAdapter
from react_baseline.runner import Budgets, run


def _state(task="V2_task_01_tomato"):
    a = EnvAdapter(task)
    try:
        a.reset()
        return a.state
    finally:
        a.close()


def _seen(tkg, ctx0, n):
    e = tkg.expectations.setdefault(f"exp:ctx0:{ctx0}", Expectation("ctx0", ctx0))
    e.outcomes["x"] += n


def test_prefers_the_least_tried_situation():
    state, tkg = _state(), TKG()
    pol = CheapPolicy(random.Random(0), epsilon=0.0)
    cands = pol.candidates(state)
    keys = {predicted_ctx0(c, state) for c in cands}
    target = "interact | hands:nothing | cutting board:empty"
    assert target in keys
    for k in keys - {target}:
        _seen(tkg, k, 5)
    for _ in range(20):
        c = pol.choose(state, tkg)
        assert predicted_ctx0(c, state) == target and c.text == "interact(0,3)"
    _seen(tkg, target, 9)          # now the most tried: never chosen at epsilon 0
    assert all(predicted_ctx0(pol.choose(state, tkg), state) != target for _ in range(20))


def test_candidates_are_valid_and_epsilon_explores():
    state = _state("V2_task_07_tomato_steak")
    pol = CheapPolicy(random.Random(1), epsilon=1.0)
    valid = set(valid_intents(state))
    picks = {pol.choose(state, TKG()).text for _ in range(200)}
    assert picks <= valid | {"wait(3)"} and len(picks) > 5
    kinds = {c.kind for c in pol.candidates(state)}
    assert kinds == {"interact", "wait", "goto"}


def test_cheap_agent_run_valid_intents_tkg_and_determinism(tmp_path):
    def go(d, seed):
        s = run(CheapAgent(seed), "V2_task_11_tomato_lettuce_onion", seed, Budgets(max_env_steps=400),
                tmp_path / d, {"backend": None, "model": None})
        return s, (tmp_path / d / "steps.jsonl").read_text(), (tmp_path / d / "tkg.json").read_text()
    s, steps_a, tkg_a = go("a", 0)
    _, steps_b, tkg_b = go("b", 0)
    assert steps_a == steps_b and tkg_a == tkg_b                 # same seed => identical logs and graph
    assert go("c", 1)[1] != steps_a
    assert s["llm_calls"] == 0 and s["intent_failures"] == 0 and s["tkg_nodes"] > 0
    rows = [json.loads(l) for l in (tmp_path / "a" / "transitions.jsonl").read_text().splitlines()]
    assert all(r["actor"] == "cheap" and r["kind"] in ("interact", "wait", "goto") for r in rows)
    assert all(r["error"] is None for r in rows)
    graph = json.loads(tkg_a)
    assert sum(n["type"] == "Transition" for n in graph["nodes"]) == len(rows) == s["tkg_transitions"]
