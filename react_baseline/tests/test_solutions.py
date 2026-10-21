"""Scripted correct-play solutions and normal-play bug IDs (post hoc)."""
import pytest

from react_baseline.agents import RandomAgent
from react_baseline.runner import Budgets, run
from react_baseline.solutions import load_normal_play, run_solution, solution_tasks

GRID_TASKS = ["V2_task_01_tomato", "V2_task_07_tomato_steak", "V2_task_11_tomato_lettuce_onion"]


def test_grid_tasks_have_solutions():
    assert set(GRID_TASKS) <= set(solution_tasks())


@pytest.mark.parametrize("task", solution_tasks())
def test_solution_delivers_and_normal_play_file_in_sync(task):
    r = run_solution(task)
    assert r["delivered"], "\n".join(r["traces"])
    assert load_normal_play()[task] == r["bug_ids_raw"], \
        "configs/normal_play_bugs.yaml is stale: run python -m react_baseline.solutions --write"


def test_summary_excludes_normal_play_but_changes_nothing_else(tmp_path):
    s = run(RandomAgent(0), "V2_task_07_tomato_steak", 0, Budgets(max_env_steps=300), tmp_path,
            {"backend": None, "model": None})
    normal = set(load_normal_play()["V2_task_07_tomato_steak"])
    assert s["unique_bugs_excluding_normal_play"] == len(set(s["unique_bug_ids"]) - normal)
    assert s["ignore_bug_ids"] == [5] and s["n_unique_bugs"] == len(s["unique_bug_ids"])
    s2 = run(RandomAgent(0), "V2_task_04_steak", 0, Budgets(max_env_steps=10), tmp_path / "b",
             {"backend": None, "model": None})
    assert s2["unique_bugs_excluding_normal_play"] is None   # no solution for this task
