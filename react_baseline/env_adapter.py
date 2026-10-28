"""Thin adapter around the SAGE Overcooked Plus env.

Responsibilities:
  * make the vendored env importable (sys.path + headless SDL + warning filters),
  * render the env state as compact text for an LLM (``render_text``),
  * split each step into what the agent may see (``obs_text``) and the hidden
    oracle (bug predicate IDs, raw item log) that only the logger may see.

Coordinate convention (verified by tests/test_env_adapter.py):
  The env stores positions as ``item.x`` = ROW and ``item.y`` = COLUMN, and the
  grid is indexed ``map[item.x][item.y]``. Action 0 ("right") increments
  ``item.y``; action 1 ("down") increments ``item.x``. Everything this module
  shows the agent uses the conventional screen frame instead:
      x = column (grows to the right), y = row (grows downward),
  i.e. screen (x, y) == env (item.y, item.x).
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# gym 0.26 prints a "Gym has been unmaintained ... NumPy 2.0" notice on import,
# and emits DeprecationWarnings for np.bool8 et al.
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"gym(\.|$)")
warnings.filterwarnings("ignore", category=UserWarning, module=r"gym(\.|$)")

REPO_ROOT = Path(__file__).resolve().parent.parent
SAGE_ROOT = REPO_ROOT / "external" / "SAGE"
SAGE_SCENARIO_DIR = SAGE_ROOT / "scenarios" / "overcooked_plus"

try:  # the env's debug_manager imports tkinter unconditionally, even headless
    import tkinter  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the interpreter
    raise ImportError(
        "tkinter is not available in this Python interpreter, but the Overcooked "
        "Plus env imports it even when headless. Install a Python 3.10 build that "
        "ships Tk (python.org installers and `uv python install 3.10` do; on "
        "Debian/Ubuntu: `apt install python3.10-tk`)."
    ) from exc

if not (SAGE_SCENARIO_DIR / "overcookedPlus").is_dir():  # pragma: no cover
    raise ImportError(
        f"SAGE env not found at {SAGE_SCENARIO_DIR}. Clone it with:\n"
        "  git clone https://github.com/BlueLinkX/SAGE external/SAGE")
if str(SAGE_SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SAGE_SCENARIO_DIR))

import io  # noqa: E402
import contextlib  # noqa: E402

with contextlib.redirect_stderr(io.StringIO()):  # swallow gym's import banner
    import gym  # noqa: E402,F401
from overcookedPlus.env_creater_new import create_env  # noqa: E402
from overcookedPlus import items as I  # noqa: E402
from overcookedPlus.constants import ITEMIDX  # noqa: E402

MAX_EPISODE_STEPS = 200  # hard-coded in OvercookedPlus.step
ACTION_NAMES = {0: "right", 1: "down", 2: "left", 3: "up", 4: "wait"}

# One distinct character per tile type. Synthesised dishes (sushi, pizza,
# roast fish) only ever exist on a plate, so they show up as 'P' on the grid.
TILE_CHARS = {
    "block": "B", "space": ".", "counter": "#", "agent": "@",
    "tomato": "T", "lettuce": "L", "onion": "O", "steak": "S", "fish": "F",
    "rice": "R", "cheese": "C", "dough": "D", "plate": "P", "knife": "K",
    "delivery": "X", "pan": "N", "sink": "W", "trash_can": "G",
    "ricecooker": "Q", "oven": "V", "sushi": "s", "pizza": "z", "roastfish": "f",
}
TILE_DESCRIPTIONS = {
    "block": "wall (impassable, cannot hold items)",
    "space": "floor (walkable)",
    "counter": "empty counter (holds one item)",
    "agent": "you",
    "plate": "plate (on a counter)",
    "knife": "cutting board",
    "delivery": "delivery window",
    "pan": "pan",
    "sink": "sink",
    "trash_can": "trash can",
    "ricecooker": "rice cooker",
    "oven": "oven",
}
STATION_NOTE = "Station letters do not show what the station holds; see STATIONS."
_REVERSE_IDX = {v: k for k, v in ITEMIDX.items()}
_FOOD_KEYS = ["tomato", "lettuce", "onion", "steak", "fish", "rice", "cheese",
              "dough", "sushi", "pizza", "roastfish"]
_DEVICE_KEYS = ["knife", "pan", "oven", "ricecooker", "sink", "delivery", "trash_can"]


def make_env(task_id: str = "V2_task_01_tomato"):
    """Create the gym-wrapped env with bug predicates enabled."""
    return create_env(task_id=task_id, check_bug=True, need_Graph=False,
                      disable_env_checker=True)


def list_tasks() -> list[str]:
    import json
    return json.loads((SAGE_SCENARIO_DIR / "tasks.json").read_text())


# ----------------------------------------------------------------------------
# Rendering
def _pos(item) -> tuple[int, int]:
    """Env (row, col) -> screen (x=col, y=row)."""
    return int(item.y), int(item.x)


def _fmt_pos(item) -> str:
    x, y = _pos(item)
    return f"({x},{y})"


# Every Food has required_chopped_times=1 by default, but only these classes are
# accepted by Knife.holdable_list, so only they have a reachable chop state.
_CHOPPABLE = (I.Tomato, I.Lettuce, I.Onion, I.Steak, I.Fish)


def _food_desc(item) -> str:
    states = []
    if isinstance(item, _CHOPPABLE) and item.is_need_chop:
        states.append("chopped" if item.chopped else "unchopped")
    if item.is_need_cook:
        states.append("cooked" if item.cooked else "uncooked")
    if getattr(item, "burned", False):
        states.append("burned")
    return f"{item.rawName} [{', '.join(states)}]" if states else item.rawName


def _plate_desc(plate) -> str:
    state = "dirty" if plate.dirty else "clean"
    contents = ", ".join(_food_desc(f) for f in plate.containing) or "empty"
    return f"plate [{state}; contents: {contents}]"


def describe_item(item) -> str:
    if item is None:
        return "nothing"
    if isinstance(item, I.Plate):
        return _plate_desc(item)
    if isinstance(item, I.Food):
        return _food_desc(item)
    return getattr(item, "rawName", type(item).__name__)


def _device_desc(dev) -> str:
    name = TILE_DESCRIPTIONS.get(dev.rawName, dev.rawName)
    if dev.rawName in ("delivery", "trash_can"):
        return name
    held = dev.holding
    if held is None:
        return f"{name}: empty"
    progress = ""
    if dev.rawName == "knife" and isinstance(held, I.Food) and held.is_need_chop:
        progress = f" (chop {min(held.cur_chopped_times, held.required_chopped_times)}/{held.required_chopped_times})"
    elif dev.rawName in ("pan", "oven", "ricecooker") and isinstance(held, I.Food):
        req = max(held.required_cooked_times, 1)
        progress = f" (cook {min(held.cur_cooked_times, req)}/{req})"
        if getattr(dev, "burned_able", False) and hasattr(held, "cur_burned_times"):
            progress += f" (burn {held.cur_burned_times}/{held.required_burned_times})"
    elif dev.rawName == "sink" and isinstance(held, I.Plate):
        progress = f" (wash {min(held.cur_wash_times, held.required_wash_times)}/{held.required_wash_times})"
    return f"{name}: holding {describe_item(held)}{progress}"


def render_text(env, last_event: str | None = None) -> str:
    """Compact, LLM-readable view of the env state.

    Built only from game state (map, items, agent, orders, step counter); it
    never reads ``info["bugs"]`` or the bug manager.
    """
    u = env.unwrapped
    im = u.item_Manager
    agent = u.agent[0]
    grid = u.map  # grid[row][col]
    n_rows, n_cols = len(grid), len(grid[0])

    used = set()
    lines = ["MAP (x = column, y = row; (0,0) is top-left; right = x+1, down = y+1)"]
    lines.append("    " + " ".join(str(c) for c in range(n_cols)))
    for r in range(n_rows):
        row_chars = []
        for c in range(n_cols):
            name = _REVERSE_IDX.get(grid[r][c], "?")
            used.add(name)
            row_chars.append(TILE_CHARS.get(name, "?"))
        lines.append(f"{r:>3} " + " ".join(row_chars))
    legend = []
    for name in TILE_CHARS:
        if name in used:
            legend.append(f"{TILE_CHARS[name]}={TILE_DESCRIPTIONS.get(name, name + ' (on a counter)')}")
    lines.append("Legend: " + ", ".join(legend))
    lines.append(STATION_NOTE)

    # Who holds what, so every item is listed once with its true location.
    holder = {}
    for key in _DEVICE_KEYS:
        for dev in im.itemDic.get(key, []):
            if getattr(dev, "holding", None) is not None:
                holder[id(dev.holding)] = f"in {TILE_DESCRIPTIONS.get(dev.rawName, dev.rawName)} at {_fmt_pos(dev)}"
    for plate in im.plate:
        for f in plate.containing:
            holder[id(f)] = "on plate"
    if agent.holding is not None:
        holder[id(agent.holding)] = "held by you"

    lines.append("")
    lines.append("STATIONS")
    for key in _DEVICE_KEYS:
        for dev in im.itemDic.get(key, []):
            lines.append(f"- {_fmt_pos(dev)} {_device_desc(dev)}")

    lines.append("ITEMS")
    item_lines = []
    def item_line(item, desc):
        where = holder.get(id(item), "on counter")
        if where == "held by you":  # it moves with you: no coordinate (see the YOU line)
            return f"- {desc} ({where})"
        return f"- {_fmt_pos(item)} {desc} ({where})"

    for plate in im.plate:
        if getattr(plate, "consumed", False) or plate.x < 0:
            continue
        item_lines.append(item_line(plate, _plate_desc(plate)))
    for key in _FOOD_KEYS:
        for food in im.itemDic.get(key, []):
            if getattr(food, "consumed", False) or food.x < 0:
                continue
            where = holder.get(id(food), "on counter")
            if where == "on plate" or where.startswith("in "):
                continue  # already listed with its plate / station
            item_lines.append(item_line(food, _food_desc(food)))
    lines.extend(item_lines or ["- (none)"])

    lines.append("")
    orders = "; ".join(" + ".join(t["ingredients"]) for t in u.tasks) or "(none)"
    lines.append(f"ORDER: {orders}")
    lines.append(f"YOU: at {_fmt_pos(agent)}, holding {describe_item(agent.holding)}")
    lines.append("ADJACENT (one move away; moving into a non-floor tile interacts with it "
                 "and you stay in place):")
    lines.extend(_adjacent_lines(u, agent))
    step = int(u.env_step)
    lines.append(f"STEP: {step}/{MAX_EPISODE_STEPS} in this episode ({MAX_EPISODE_STEPS - step} remaining)")
    if last_event:
        lines.append(f"LAST EVENT: {last_event}")
    return "\n".join(lines)


_ENV_DELTAS = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}  # env (row, col), = constants.DIRECTION


def _tile_desc(u, row: int, col: int) -> str:
    """What occupies env tile (row, col): perception only, no outcome prediction."""
    grid = u.map
    if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
        return "edge of the map"
    name = _REVERSE_IDX.get(grid[row][col], "?")
    if name == "space":
        return "floor (you walk there)"
    im = u.item_Manager
    for key in _DEVICE_KEYS:
        for dev in im.itemDic.get(key, []):
            if dev.x == row and dev.y == col:
                return _device_desc(dev)
    if name == "plate":
        for plate in im.plate:
            if plate.x == row and plate.y == col and plate is not u.agent[0].holding:
                return _plate_desc(plate) + " on a counter"
    if name in _FOOD_KEYS:  # the tile code says this food type sits loose on the counter
        for food in im.itemDic.get(name, []):
            if food.x == row and food.y == col and not getattr(food, "consumed", False):
                return _food_desc(food) + " on a counter"
    if name == "agent":
        return "another chef"
    return TILE_DESCRIPTIONS.get(name, name)


def _adjacent_lines(u, agent) -> list[str]:
    out = []
    for action in (0, 1, 2, 3):
        dr, dc = _ENV_DELTAS[action]
        row, col = agent.x + dr, agent.y + dc
        out.append(f"- {action} {ACTION_NAMES[action]:<5} -> ({col},{row}): {_tile_desc(u, row, col)}")
    return out


# ----------------------------------------------------------------------------
# Structured agent-visible state (for the Action reader and, later, TKG memory)
def _item_record(item) -> dict | None:
    """Structured form of what ``describe_item`` shows the agent."""
    if item is None:
        return None
    rec = {"name": getattr(item, "rawName", type(item).__name__)}
    if isinstance(item, I.Plate):
        rec["dirty"] = bool(item.dirty)
        rec["contents"] = [_item_record(f) for f in item.containing]
    elif isinstance(item, I.Food):
        if isinstance(item, _CHOPPABLE) and item.is_need_chop:
            rec["chopped"] = bool(item.chopped)
        if item.is_need_cook:
            rec["cooked"] = bool(item.cooked)
        if getattr(item, "burned", False):
            rec["burned"] = True
    return rec


def _station_progress(dev) -> str | None:
    """The '(chop 0/1)'-style progress the renderer prints, without parentheses."""
    desc = _device_desc(dev)
    if "(" in desc and desc.endswith(")"):
        return desc[desc.rindex("(") + 1:-1]
    return None


def _tile_record(u, row: int, col: int) -> dict:
    """What env tile (row, col) visibly holds: {"kind": display name, "holding": item record}."""
    grid = u.map
    if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
        return {"kind": "edge of the map", "holding": None}
    im, agent = u.item_Manager, u.agent[0]
    for key in _DEVICE_KEYS:
        for dev in im.itemDic.get(key, []):
            if dev.x == row and dev.y == col:
                return {"kind": display_name(dev.rawName), "holding": _item_record(getattr(dev, "holding", None))}
    name = _REVERSE_IDX.get(grid[row][col], "?")
    if name == "space" or (name == "agent" and (agent.x, agent.y) == (row, col)):
        return {"kind": "floor", "holding": None}
    if name == "agent":
        return {"kind": "another chef", "holding": None}
    held = agent.holding
    on_plate = {id(f) for plate in im.plate for f in plate.containing}
    for plate in im.plate:
        if plate.x == row and plate.y == col and plate is not held and not getattr(plate, "consumed", False):
            return {"kind": "counter", "holding": _item_record(plate)}
    for key in _FOOD_KEYS:
        for food in im.itemDic.get(key, []):
            if (food.x == row and food.y == col and food is not held and id(food) not in on_plate
                    and not getattr(food, "consumed", False)):
                return {"kind": "counter", "holding": _item_record(food)}
    if name in ("counter", "plate") or name in _FOOD_KEYS:
        return {"kind": "counter", "holding": None}
    return {"kind": display_name(name), "holding": None}


def step_context(env, action: int) -> dict:
    """Agent-visible facts from just before a step: what you hold, and the tile the action
    is aimed at (screen [x, y]) with what it holds. ``finish_context`` adds the after-state."""
    u = env.unwrapped
    agent = u.agent[0]
    ctx = {"action": int(action), "held_before": _item_record(agent.holding),
           "tile": None, "tile_before": None, "_env_tile": None}
    if int(action) in _ENV_DELTAS:
        dr, dc = _ENV_DELTAS[int(action)]
        row, col = int(agent.x) + dr, int(agent.y) + dc
        ctx.update(tile=[col, row], tile_before=_tile_record(u, row, col), _env_tile=(row, col))
    return ctx


def finish_context(env, ctx: dict) -> dict:
    u = env.unwrapped
    out = {k: v for k, v in ctx.items() if k != "_env_tile"}
    out["held_after"] = _item_record(u.agent[0].holding)
    out["tile_after"] = _tile_record(u, *ctx["_env_tile"]) if ctx.get("_env_tile") else None
    return out


def event_record(info, ctx: dict | None = None) -> dict | None:
    """Agent-visible last event: type, target, success flag and the rendered text. With the
    step context (``step_context`` + ``finish_context``) it also carries held_before,
    held_after, tile ([x, y] the action was aimed at), tile_before and tile_after."""
    if info is None:
        return None
    ev = _first_event(info)
    et = ev.get("event_type")
    et = None if et in ("None", None) else str(et)
    target = ev.get("target")
    if et == "move" or not isinstance(target, (str, list)) or not target:
        target = None  # move targets are in env (row, col) order; the position is in "agent"
    elif isinstance(target, list):
        target = [display_name(str(t)) for t in target]
    else:
        target = display_name(target)
    success = (ev.get("details", {}).get("others", {}) or {}).get("success")
    rec = {"type": et, "target": target,
           "success": success if isinstance(success, bool) else None,
           "text": describe_event(info, ctx)}
    if ctx is not None:
        rec.update({k: ctx.get(k) for k in ("held_before", "held_after", "tile", "tile_before", "tile_after")})
    return rec


def visible_state(env, info=None, event: dict | None = None) -> dict:
    """Agent-visible state as a dict, built from the same game state as
    ``render_text`` (screen frame: pos = [x=column, y=row]). Never reads
    ``info["bugs"]`` or the bug manager. ``event``: a ready ``event_record``."""
    u = env.unwrapped
    im = u.item_Manager
    agent = u.agent[0]
    grid = u.map
    rows = ["".join(TILE_CHARS.get(_REVERSE_IDX.get(code, "?"), "?") for code in row) for row in grid]

    held_by_station = {}
    stations = []
    for key in _DEVICE_KEYS:
        for dev in im.itemDic.get(key, []):
            if getattr(dev, "holding", None) is not None:
                held_by_station[id(dev.holding)] = dev
            stations.append({
                "kind": dev.rawName,
                "label": TILE_DESCRIPTIONS.get(dev.rawName, dev.rawName),
                "pos": list(_pos(dev)),
                "holding": _item_record(getattr(dev, "holding", None)),
                "progress": _station_progress(dev),
            })
    on_plate = {id(f) for plate in im.plate for f in plate.containing}
    held = agent.holding

    def where(item):
        if item is held:
            return "held"
        if id(item) in held_by_station:
            return "station"
        if id(item) in on_plate:
            return "plate"
        return "counter"

    items = []
    for plate in im.plate:
        if getattr(plate, "consumed", False) or plate.x < 0:
            continue
        items.append({**_item_record(plate), "pos": list(_pos(plate)), "where": where(plate)})
    for key in _FOOD_KEYS:
        for food in im.itemDic.get(key, []):
            if getattr(food, "consumed", False) or food.x < 0:
                continue
            w = where(food)
            if w in ("plate", "station"):
                continue  # listed with its plate / station, as in render_text
            items.append({**_item_record(food), "pos": list(_pos(food)), "where": w})

    return {
        "episode_step": int(u.env_step),
        "map": rows,
        "agent": {"pos": list(_pos(agent)), "holding": _item_record(held)},
        "items": items,
        "stations": stations,
        "order": [list(t["ingredients"]) for t in u.tasks],
        "last_event": event if event is not None else event_record(info),
    }


def summarize_state(env, last_event: str | None = None) -> str:
    """One-line observation summary used in the agent's rolling history."""
    agent = env.unwrapped.agent[0]
    s = f"you at {_fmt_pos(agent)} holding {describe_item(agent.holding)}"
    if last_event:
        s += f"; event: {last_event}"
    return s


# ----------------------------------------------------------------------------
# Stepping with an oracle split
def _first_event(info) -> dict:
    log = info.get("item_log") or []
    return log[0] if log and isinstance(log[0], dict) else {}


def delivery_success(info) -> bool:
    return any(e.get("event_type") == "deliver"
               and e.get("details", {}).get("others", {}).get("success") is True
               for e in info.get("item_log", []) if isinstance(e, dict))


# Event targets come from SAGE's internal names ("knife", "trash_can", ...). Show the
# agent the same names the map legend and STATIONS list use ("cutting board", ...).
_DISPLAY_KEYS = set(_DEVICE_KEYS) | {"block"}


def display_name(raw: str) -> str:
    if raw == "block":
        return "wall"
    return TILE_DESCRIPTIONS.get(raw, raw) if raw in _DISPLAY_KEYS else raw


def item_phrase(rec: dict | None) -> str:
    """Short player-view name of an item record: "chopped tomato", "the plate with [...]"."""
    if rec is None:
        return "nothing"
    if rec.get("name") == "plate":
        contents = [item_phrase(c) for c in rec.get("contents") or []]
        if rec.get("dirty"):
            return "the dirty plate" + (f" with [{', '.join(contents)}]" if contents else "")
        return f"the plate with [{', '.join(contents)}]" if contents else "the empty plate"
    states = [s for s in ("chopped", "cooked", "burned") if rec.get(s)]
    return " ".join(states + [rec.get("name", "item")])


def _plate_contents(rec: dict | None) -> str:
    if not rec or rec.get("name") != "plate":
        return item_phrase(rec)
    return "[" + ", ".join(item_phrase(c) for c in rec.get("contents") or []) + "]"


_PUT_IN = {"pan", "oven", "rice cooker", "sink", "trash can"}   # "in the pan", else "on the ..."
_TAKE_OUT = {"pan", "oven", "rice cooker", "sink"}              # "took X out of the pan"


def describe_event(info, ctx: dict | None = None) -> str:
    """Player-visible feedback for the last action (no bug information).

    Says what happened from the player's view: what you held before / after, the
    tile you acted on (by its legend name and position) and what it now holds. It
    never explains why. item_log entries carry ``event_type`` and an ``others`` dict
    with a success flag; the bug IDs were already popped out of them by the env's
    logger. ``ctx`` is the step context (``step_context`` + ``finish_context``).
    """
    ev = _first_event(info)
    et = ev.get("event_type", "None")
    success = (ev.get("details", {}).get("others", {}) or {}).get("success")
    target = ev.get("target")
    c = ctx or {}
    hb, ha = c.get("held_before"), c.get("held_after")
    tb, ta = c.get("tile_before") or {}, c.get("tile_after") or {}
    tb_item, ta_item = tb.get("holding"), ta.get("holding")
    kind = tb.get("kind") or (display_name(target) if isinstance(target, str) and target else None)
    at = f" at ({c['tile'][0]},{c['tile'][1]})" if c.get("tile") else ""
    the_tile = f"the {kind}{at}" if kind else "it"
    prep = "in" if kind in _PUT_IN else "on"

    def outcome() -> str:  # after an attempt the game flags as failed: what (if anything) changed
        changes = []
        if hb != ha:
            changes.append(f"you now hold {item_phrase(ha)}")
        if tb_item != ta_item:
            changes.append(f"{the_tile} now holds {item_phrase(ta_item)}")
        return "; ".join(changes) or "nothing changed"

    if et == "move":
        return "moved"
    if et in ("None", None):
        if c.get("action") == 4:
            return "waited"
        if not kind:
            return "nothing happened"
        what = kind
        if kind == "counter":
            what = f"{item_phrase(tb_item)} on the counter" if tb_item else "empty counter"
        bumped = f"bumped {what}{at}"
        changed = outcome()
        return f"nothing happened ({bumped})" if changed == "nothing changed" else f"{bumped}; {changed}"
    if et == "deliver":
        return f"delivered {item_phrase(hb)}: order {'ACCEPTED' if success else 'REJECTED'}"
    if et == "pickup":
        if ha is None:
            return f"tried to pick something up from {the_tile}: {outcome()}"
        if kind in _TAKE_OUT:
            return f"took {item_phrase(ha)} out of {the_tile}"
        return f"picked up {item_phrase(ha)} from {the_tile}"
    if et == "putdown":
        onto_plate = target == "plate" or (tb_item or {}).get("name") == "plate"
        if success:
            if onto_plate:
                return f"put {item_phrase(hb)} onto the plate{at}; the plate now holds {_plate_contents(ta_item)}"
            if kind == "counter":
                return f"put {item_phrase(hb)} down on the counter{at}"
            return f"put {item_phrase(hb)} {prep} {the_tile}"
        if onto_plate:
            return f"tried to put {item_phrase(hb)} onto the plate{at}: {outcome()}"
        if kind == "counter" and tb_item:
            return f"tried to put {item_phrase(hb)} down{at}, where the {item_phrase(tb_item)} is: {outcome()}"
        if kind == "counter":
            return f"tried to put {item_phrase(hb)} down on the counter{at}: {outcome()}"
        return f"tried to put {item_phrase(hb)} {prep} {the_tile}: {outcome()}"
    if et == "pickup_into_plate":
        src = f"the {kind}{at}" if kind else "there"
        if success:
            return (f"added {item_phrase(tb_item)} from {src} to the plate you are holding; "
                    f"the plate now holds {_plate_contents(ha)}")
        if tb_item:
            return f"tried to add {item_phrase(tb_item)} from {src} to the plate you are holding: {outcome()}"
        return f"tried to add from the empty {kind or 'tile'}{at} to the plate you are holding: {outcome()}"
    if et == "chop":
        if success is False:
            return f"tried to chop {item_phrase(tb_item)} on {the_tile}: {outcome()}"
        return f"chopped {item_phrase(tb_item)} on {the_tile}"
    if et == "wash":
        state = ""
        if (ta_item or {}).get("name") == "plate":
            state = " (still dirty)" if ta_item.get("dirty") else " (now clean)"
        return f"scrubbed the plate in {the_tile}{state}"
    if et == "dump":
        text = f"threw away {item_phrase(hb)} in {the_tile}"
        return text + (f"; you now hold {item_phrase(ha)}" if ha is not None else "")
    return f"interacted with {the_tile}: {outcome()}"


@dataclass
class StepResult:
    obs_text: str           # agent-visible
    summary: str            # agent-visible one-liner for history
    reward: float
    done: bool
    episode_step: int
    oracle: dict = field(default_factory=dict)  # hidden: never shown to the agent
    state: dict = field(default_factory=dict)   # agent-visible structured state


class EnvAdapter:
    """Holds one env and exposes agent-visible text + hidden oracle per step."""

    def __init__(self, task_id: str, seed: int | None = None):
        self.task_id = task_id
        self.env = make_env(task_id)
        if seed is not None:
            self.env.action_space.seed(seed)
        self.state: dict = {}

    def reset(self) -> str:
        self.env.reset()
        self.state = visible_state(self.env)
        return render_text(self.env)

    def step(self, action: int) -> StepResult:
        ctx = step_context(self.env, action)
        _, reward, done, info = self.env.step(int(action))
        rec = event_record(info, finish_context(self.env, ctx))
        event = rec["text"]
        self.state = visible_state(self.env, info, event=rec)
        return StepResult(
            obs_text=render_text(self.env, last_event=event),
            summary=summarize_state(self.env, last_event=event),
            reward=float(reward),
            done=bool(done),
            episode_step=int(self.env.unwrapped.env_step),
            oracle={
                "bugs": list(info.get("bugs", [])),
                "delivery_success": delivery_success(info),
                "event_type": _first_event(info).get("event_type"),
            },
            state=self.state,
        )

    def close(self):
        self.env.close()
