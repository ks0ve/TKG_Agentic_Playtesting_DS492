import json
import os
import re
from pathlib import Path

import pytest

from react_baseline.env_adapter import (EnvAdapter, SAGE_SCENARIO_DIR, TILE_CHARS, make_env,
                                        render_text)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
DEMOS = json.loads((SAGE_SCENARIO_DIR / "verified_demonstrations.json").read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("demo", DEMOS, ids=[d["task"] for d in DEMOS])
def test_verified_demonstrations_deliver(demo):
    adapter = EnvAdapter(demo["task"])
    try:
        obs = adapter.reset()
        assert "YOU:" in obs
        res = None
        for action in demo["actions"]:
            res = adapter.step(action)  # renders every step
            assert "MAP" in res.obs_text and "YOU:" in res.obs_text
        assert res.done
        assert res.oracle["delivery_success"] is True
        assert "ACCEPTED" in res.obs_text
        assert res.episode_step == demo["steps"]
    finally:
        adapter.close()


def _you(text):
    m = re.search(r"YOU: at \((\d+),(\d+)\)", text)
    return int(m.group(1)), int(m.group(2))


def _at_in_grid(text):
    grid_lines = text.split("\n")[2:]
    for y, line in enumerate(grid_lines):
        if not re.match(r"^\s*\d+ ", line):
            break
        cells = line.split()[1:]
        if "@" in cells:
            return cells.index("@"), y
    raise AssertionError("agent not on grid")


@pytest.mark.parametrize("action,dx,dy", [(0, 1, 0), (1, 0, 1)])
def test_coordinate_convention(action, dx, dy):
    """A scripted move changes the position exactly as the renderer claims:
    x = column (action 0 = right = x+1), y = row (action 1 = down = y+1)."""
    adapter = EnvAdapter("V2_task_01_tomato")
    try:
        before = adapter.reset()
        x0, y0 = _you(before)
        assert _at_in_grid(before) == (x0, y0)
        u = adapter.env.unwrapped
        assert u.map[y0][x0] == 2  # agent tile code at map[row][col]
        after = adapter.step(action).obs_text
        assert _you(after) == (x0 + dx, y0 + dy)
        assert _at_in_grid(after) == (x0 + dx, y0 + dy)
        # env-internal convention: item.x is the ROW, item.y is the COLUMN
        assert (u.agent[0].y, u.agent[0].x) == (x0 + dx, y0 + dy)
    finally:
        adapter.close()


def test_legend_chars_distinct():
    assert len(set(TILE_CHARS.values())) == len(TILE_CHARS)


@pytest.mark.parametrize("task", ["V2_task_01_tomato", "V2_task_04_steak", "V2_task_78_onion_fish_rice"])
def test_render_snapshot(task):
    env = make_env(task)
    try:
        env.reset()
        text = render_text(env)
    finally:
        env.close()
    snap = SNAPSHOT_DIR / f"{task}.txt"
    if os.environ.get("UPDATE_SNAPSHOTS") or not snap.exists():
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        snap.write_text(text + "\n", encoding="utf-8")
    assert text + "\n" == snap.read_text(encoding="utf-8")


def test_adjacent_block_matches_moves():
    """ADJACENT lists what is one move away, in screen coordinates, per action."""
    adapter = EnvAdapter("V2_task_01_tomato")
    try:
        text = adapter.reset()
        assert "- 0 right -> (2,1): floor (you walk there)" in text
        assert "- 1 down  -> (1,2): floor (you walk there)" in text
        assert "- 2 left  -> (0,1): steak [unchopped, uncooked] on a counter" in text
        assert "- 3 up    -> (1,0): empty counter" in text
        for a in (1, 1):  # to (1,3), where run #1 got stuck
            text = adapter.step(a).obs_text
        assert "- 1 down  -> (1,4): trash can" in text
        assert "- 2 left  -> (0,3): cutting board: empty" in text
    finally:
        adapter.close()
