"""Order-independent normal-play sets from the recipe planner (configs/normal_play_bugs.yaml)."""
import math

import pytest

from react_baseline.normal_play import (GRID_TASKS, load_yaml, plan_task, play_variant, recipe, variant_name,
                                        variants)
from react_baseline.solutions import load_normal_play

DATA = load_yaml()["tasks"]


def test_grid_tasks_come_from_the_planner_and_every_variant_delivers():
    for task in GRID_TASKS:
        e = DATA[task]
        assert e["source"] == "planner" and e["variants"] == e["variants_delivered"] == len(e["per_variant"])
        assert e["variants"] == len(variants(e["ingredients"]))
        lists = [v for v in e["per_variant"].values()]
        assert all(isinstance(v, list) for v in lists)   # no FAILED variant
        assert e["bug_ids_raw"] == sorted({b for v in lists for b in v}) == load_normal_play()[task]
        assert e["always"] == sorted(set.intersection(*(set(v) for v in lists)))
        assert e["id_variant_counts"] == {b: sum(b in v for v in lists) for b in e["bug_ids_raw"]}


def test_variant_counts():
    # interleaved: n! assembly orders x 3^n methods; prep-first: n! prep orders x n! assembly orders x
    # 2^n counter methods (3 for a cooked ingredient left in its station), x2 plate policies where
    # two scoops follow each other
    assert len(variants(["tomato"])) == 3 + 2
    # 2 x 9 interleaved + 2 x 2 x (2 x 3) prep-first + 4 "park" twins of scoop,scoop
    assert len(variants(["tomato", "steak"])) == 18 + 24 + 4
    n3 = [v for v in variants(["tomato", "lettuce", "onion"]) if v["schedule"] == "interleaved"]
    assert len(n3) == math.factorial(3) * 27
    assert len({variant_name(v) for v in variants(["tomato", "lettuce", "onion"])}) == 558


def test_tomato_planner_matches_file_exactly():
    res = plan_task("V2_task_01_tomato")
    e = DATA["V2_task_01_tomato"]
    assert res["union"] == e["bug_ids_raw"] == [5]
    assert {r["name"]: r["bug_ids_raw"] for r in res["rows"]} == e["per_variant"]


@pytest.mark.parametrize("task", GRID_TASKS[1:])
def test_sampled_variants_replay_as_recorded(task):
    """Deterministic: a spread of variants re-played on the game gives the recorded IDs."""
    vs = variants(recipe(task))
    for v in vs[:: max(len(vs) // 12, 1)]:
        r = play_variant(task, v)
        assert r["delivered"] and r["failed_events"] == 0, r["error"]
        assert DATA[task]["per_variant"][r["name"]] == r["bug_ids_raw"], r["name"]


def test_known_order_dependent_ids_are_normal_play():
    """Bug 78 (steak first, then tomato) and the salad's 67 / 71 / 107 fire in correct play."""
    assert {4, 6, 68, 78, 127} <= set(load_normal_play()["V2_task_07_tomato_steak"])
    salad = set(load_normal_play()["V2_task_11_tomato_lettuce_onion"])
    assert {3, 4, 7, 70, 73, 74, 86, 87, 88, 102, 103, 109, 67, 71, 107} <= salad
    per = DATA["V2_task_07_tomato_steak"]["per_variant"]
    assert 78 in per["interleaved; plate steak:carry,tomato:carry"]
    assert 78 not in per["interleaved; plate tomato:carry,steak:carry"]


def test_no_washing_in_correct_play():
    """Plates start clean and a correct delivery ends the episode: no correct path washes a plate,
    so 47 (SINK_HOLD_AFTER_CLEAN) is not normal play on the grid tasks."""
    from react_baseline.env_adapter import EnvAdapter
    for task in GRID_TASKS:
        a = EnvAdapter(task)
        try:
            a.reset()
            assert all(not it["dirty"] for it in a.state["items"] if it["name"] == "plate")
        finally:
            a.close()
        assert 47 not in load_normal_play()[task]
