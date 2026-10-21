"""Action reader: the shared, deterministic, LLM-free action interface.

Every agent (ReAct in both action spaces, the random baselines, and later the
TKG arm's cheap policy and LLM) hands the reader an *intent*; the reader turns it
into button presses 0-4 and runs them through a ``step_fn`` supplied by the
runner. It contains no agent logic: no choice of what to do, only how to do it.

Intent language (coordinates in the screen frame the agent sees: x = column,
y = row; see env_adapter for the env's transposed internals):

    right | down | left | up | 0 | 1 | 2 | 3   one primitive move
    wait | 4 | stay | wait(n)                   n (1..20) wait steps
    goto(x,y)                                   shortest path over floor tiles to (x,y)
    interact(x,y) | interact(<name>)            walk next to (x,y), then bump into it once;
                                                a station/item name must resolve to one tile

Paths are breadth-first over floor tiles, expanding neighbours in the order
right, down, left, up (= action IDs 0..3), so ties go to the first path found.

Floor/non-floor tolerance: ``goto`` aimed at a non-floor tile walks to the
nearest floor tile next to it and stops without bumping (it never performs an
interaction the agent did not ask for); ``interact`` aimed at a floor tile walks
onto it. Both are reported in the trace note and logged as ``coerced``.

The reader only reads the agent-visible state dict (``env_adapter.visible_state``)
and the agent-visible ``VisibleStep`` the runner returns; it never sees the oracle.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

WAIT = 4
MOVE_NAMES = {0: "right", 1: "down", 2: "left", 3: "up"}
ACTION_NAMES = {**MOVE_NAMES, 4: "wait"}
_NAME_TO_MOVE = {"right": 0, "down": 1, "left": 2, "up": 3}
SCREEN_DELTAS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}  # screen (dx, dy) per action
FLOOR_CHARS = {".", "@"}
DEFAULT_MAX_MACRO_LEN = 40
MAX_WAIT = 20

STATION_ALIASES = {
    "knife": ("knife", "cutting board", "chopping board", "board"),
    "delivery": ("delivery", "delivery window", "window"),
    "trash_can": ("trash", "trash can", "bin"),
    "pan": ("pan", "stove"),
    "sink": ("sink",),
    "ricecooker": ("rice cooker", "ricecooker", "cooker"),
    "oven": ("oven",),
}


# ----------------------------------------------------------------------------
# Parsing
@dataclass
class Intent:
    kind: str                 # move | wait | goto | interact | invalid
    text: str                 # canonical form, e.g. "goto(3,1)"
    action: int | None = None
    n: int = 1
    x: int | None = None
    y: int | None = None
    name: str | None = None
    error: str | None = None  # parse error (kind == "invalid")


_STRIP = "*_`'\"[] \t.:"
_GOTO = re.compile(r"^(?:go[\s_-]*to|move[\s_-]*to|walk[\s_-]*to)\s*[\(\[]?\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*[\)\]]?",
                   re.IGNORECASE)
_INTERACT_XY = re.compile(r"^(?:interact|use)(?:\s+with)?\s*[\(\[]?\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*[\)\]]?",
                          re.IGNORECASE)
_INTERACT_NAME = re.compile(r"^(?:interact|use)(?:\s+with)?\s*(?:\(\s*([a-z][a-z _-]*?)\s*\)|"
                            r"\s(?:the\s+)?([a-z][a-z_-]*(?:\s+[a-z][a-z_-]*)?))", re.IGNORECASE)
_WAIT = re.compile(r"^(?:wait|stay)\b(?:\s*[\(\[]?\s*(\d+)\s*[\)\]]?)?", re.IGNORECASE)
_MOVE = re.compile(r"^(right|down|left|up)\b", re.IGNORECASE)
_DIGIT = re.compile(r"^([0-4])(?![\d,])")
_SPLIT = re.compile(r"\s*(?:;|\n|\bthen\b)\s*", re.IGNORECASE)


def parse_intent(segment: str) -> Intent:
    """Parse one intent. Tolerates case, markdown, backticks, brackets and trailing text."""
    raw = (segment or "").strip().strip(_STRIP).strip()
    if not raw:
        return Intent("invalid", raw, error="empty intent")
    if m := _GOTO.match(raw):
        x, y = int(m.group(1)), int(m.group(2))
        return Intent("goto", f"goto({x},{y})", x=x, y=y)
    if m := _INTERACT_XY.match(raw):
        x, y = int(m.group(1)), int(m.group(2))
        return Intent("interact", f"interact({x},{y})", x=x, y=y)
    if m := _INTERACT_NAME.match(raw):
        name = " ".join((m.group(1) or m.group(2)).replace("_", " ").replace("-", " ").lower().split())
        return Intent("interact", f"interact({name})", name=name)
    if m := _WAIT.match(raw):
        n = int(m.group(1)) if m.group(1) else 1
        if not 1 <= n <= MAX_WAIT:
            return Intent("invalid", f"wait({n})", error=f"wait(n) needs 1 <= n <= {MAX_WAIT}")
        return Intent("wait", "wait" if n == 1 else f"wait({n})", action=WAIT, n=n)
    if m := _MOVE.match(raw):
        a = _NAME_TO_MOVE[m.group(1).lower()]
        return Intent("move", MOVE_NAMES[a], action=a)
    if m := _DIGIT.match(raw):
        a = int(m.group(1))
        if a == WAIT:
            return Intent("wait", "wait", action=WAIT, n=1)
        return Intent("move", MOVE_NAMES[a], action=a)
    return Intent("invalid", raw[:60], error=f"unparseable intent {raw[:60]!r}")


def parse_intents(payload: str, max_n: int = 1) -> tuple[list[Intent], str | None]:
    """Split an ``Action:`` payload on ';' / newlines / 'then' and parse each part.
    Returns (intents[:max_n], note); the note says when intents were dropped."""
    parts = [p for p in _SPLIT.split(payload or "") if p.strip(_STRIP)]
    intents = [parse_intent(p) for p in parts] or [parse_intent("")]
    note = None
    if len(intents) > max_n:
        note = f"truncated {len(intents)} intents to {max_n}"
        intents = intents[:max_n]
    return intents, note


def primitive_intent(action: int) -> Intent:
    """The intent for one raw button press (used by primitive-action agents)."""
    a = int(action)
    if a == WAIT:
        return Intent("wait", "wait", action=WAIT, n=1)
    return Intent("move", MOVE_NAMES[a], action=a)


# ----------------------------------------------------------------------------
# Map helpers (visible state only)
def _is_floor(state: dict, x: int, y: int) -> bool:
    rows = state["map"]
    return 0 <= y < len(rows) and 0 <= x < len(rows[y]) and rows[y][x] in FLOOR_CHARS


def _in_map(state: dict, x: int, y: int) -> bool:
    rows = state["map"]
    return 0 <= y < len(rows) and 0 <= x < len(rows[y])


def bfs_path(state: dict, start: tuple[int, int], goals: set[tuple[int, int]]):
    """Shortest list of actions from ``start`` to the nearest goal over floor
    tiles (deterministic tie-break: neighbours expanded right, down, left, up).
    Returns (actions, goal) or (None, None) if no goal is reachable."""
    start = tuple(start)
    if start in goals:
        return [], start
    parent = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for a in (0, 1, 2, 3):
            dx, dy = SCREEN_DELTAS[a]
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in parent or not _is_floor(state, *nxt):
                continue
            parent[nxt] = (cur, a)
            if nxt in goals:
                path, node = [], nxt
                while parent[node] is not None:
                    node, act = parent[node]
                    path.append(act)
                return path[::-1], nxt
            q.append(nxt)
    return None, None


def reachable_floor(state: dict) -> list[tuple[int, int]]:
    """Floor tiles reachable from the agent, in BFS discovery order."""
    start = tuple(state["agent"]["pos"])
    seen, order, q = {start}, [start], deque([start])
    while q:
        cur = q.popleft()
        for a in (0, 1, 2, 3):
            dx, dy = SCREEN_DELTAS[a]
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt not in seen and _is_floor(state, *nxt):
                seen.add(nxt)
                order.append(nxt)
                q.append(nxt)
    return order


def resolve_name(state: dict, name: str) -> tuple[tuple[int, int] | None, str | None]:
    """Map a station/item name to its single tile. Returns (pos, error)."""
    name = name.lower().removeprefix("the ").strip()
    hits = set()
    for st in state.get("stations", []):
        aliases = STATION_ALIASES.get(st["kind"], (st["kind"],)) + (st["label"].lower(),)
        held = (st.get("holding") or {}).get("name")
        if name in aliases or name == held:
            hits.add(tuple(st["pos"]))
    for it in state.get("items", []):
        if it.get("where") in ("counter", "station") and it.get("name") == name:
            hits.add(tuple(it["pos"]))
    if not hits:
        return None, f"no visible station or item called '{name}'"
    if len(hits) > 1:
        where = ", ".join(f"({x},{y})" for x, y in sorted(hits))
        return None, f"'{name}' is ambiguous: {where}; use coordinates"
    return hits.pop(), None


def valid_intents(state: dict) -> list[str]:
    """All currently executable intents, in a fixed order: the four moves, wait,
    goto for every other reachable floor tile, interact for every non-floor tile
    next to a reachable floor tile."""
    out = [MOVE_NAMES[a] for a in (0, 1, 2, 3)] + ["wait"]
    floor = reachable_floor(state)
    me = tuple(state["agent"]["pos"])
    out += [f"goto({x},{y})" for x, y in sorted(floor) if (x, y) != me]
    targets = set()
    for fx, fy in floor:
        for dx, dy in SCREEN_DELTAS.values():
            tx, ty = fx + dx, fy + dy
            if _in_map(state, tx, ty) and not _is_floor(state, tx, ty):
                targets.add((tx, ty))
    out += [f"interact({x},{y})" for x, y in sorted(targets)]
    return out


# ----------------------------------------------------------------------------
# Execution
@dataclass
class VisibleStep:
    """What the runner hands back after one env step: agent-visible only."""
    obs_text: str
    summary: str
    state: dict
    event: str
    done: bool


@dataclass
class ExecResult:
    intent: str
    kind: str
    primitive_actions: list[int] = field(default_factory=list)
    events: list[str] = field(default_factory=list)          # rendered LAST EVENT texts
    event_records: list[dict] = field(default_factory=list)  # structured last_event per step
    stop_reason: str = ""   # completed | episode_end | budget | unexpected_event | length_cap | invalid
    error: str | None = None
    note: str | None = None
    coerced: str | None = None   # "goto->approach" | "interact->goto" when the verb was remapped
    final: VisibleStep | None = None

    @property
    def n_steps(self) -> int:
        return len(self.primitive_actions)

    @property
    def done(self) -> bool:
        return bool(self.final and self.final.done)

    @property
    def trace(self) -> str:
        """Compact agent-visible trace: steps taken, events seen, why it stopped."""
        moves = ",".join(ACTION_NAMES[a] for a in self.primitive_actions) or "none"
        ev, compact = self.events, []
        i = 0
        while i < len(ev):
            j = i
            while j + 1 < len(ev) and ev[j + 1] == ev[i]:
                j += 1
            compact.append(ev[i] if j == i else f"{ev[i]} x{j - i + 1}")
            i = j + 1
        s = f"{self.intent}: {self.n_steps} step(s) [{moves}]; events: {'; '.join(compact) or 'none'}; stop: {self.stop_reason}"
        if self.error:
            s += f"; error: {self.error}"
        if self.note:
            s += f"; note: {self.note}"
        return s

    def to_dict(self) -> dict:
        return {"intent": self.intent, "kind": self.kind, "n_steps": self.n_steps,
                "stop_reason": self.stop_reason, "error": self.error, "note": self.note,
                "coerced": self.coerced}


class ActionReader:
    def __init__(self, max_macro_len: int = DEFAULT_MAX_MACRO_LEN):
        self.max_macro_len = max_macro_len

    def plan(self, intent: Intent, state: dict):
        """-> (steps, error, note, coerced); each step is (action, expected screen pos or None).

        Floor/non-floor tolerance (same for every agent): ``goto`` on a non-floor tile
        walks next to it and stops without bumping (coerced "goto->approach");
        ``interact`` on a floor tile walks onto it (coerced "interact->goto")."""
        if intent.kind == "invalid":
            return None, intent.error, None, None
        if intent.kind == "move":
            return [(intent.action, None)], None, None, None
        if intent.kind == "wait":
            return [(WAIT, None)] * intent.n, None, None, None
        me = tuple(state["agent"]["pos"])
        if intent.kind == "goto":
            goal = (intent.x, intent.y)
            if not _in_map(state, *goal):
                return None, "outside the map", None, None
            if not _is_floor(state, *goal):
                path, _ = bfs_path(state, me, set(self._adjacent_floor(state, goal)))
                if path is None:
                    return None, "not a floor tile, and no floor tile next to it is reachable", None, None
                note = (f"({goal[0]},{goal[1]}) is not a floor tile: stopped next to it; "
                        f"use interact({goal[0]},{goal[1]}) to use it")
                if not path:
                    note = f"already next to ({goal[0]},{goal[1]}), which is not a floor tile; " \
                           f"use interact({goal[0]},{goal[1]}) to use it"
                return self._walk(me, path), None, note, "goto->approach"
            path, _ = bfs_path(state, me, {goal})
            if path is None:
                return None, "no floor path to that tile", None, None
            if not path:
                return [], None, "already there", None
            return self._walk(me, path), None, None, None
        if intent.kind == "interact":
            if intent.name is not None:
                target, err = resolve_name(state, intent.name)
                if err:
                    return None, err, None, None
            else:
                target = (intent.x, intent.y)
            if not _in_map(state, *target):
                return None, "outside the map", None, None
            if _is_floor(state, *target):
                path, _ = bfs_path(state, me, {target})
                if path is None:
                    return None, f"({target[0]},{target[1]}) is a floor tile with no path to it", None, None
                note = f"({target[0]},{target[1]}) is a floor tile: walked onto it"
                if not path:
                    note = f"({target[0]},{target[1]}) is a floor tile and you are on it"
                return self._walk(me, path), None, note, "interact->goto"
            adj = self._adjacent_floor(state, target)
            path, spot = bfs_path(state, me, set(adj))
            if path is None:
                return None, "no floor tile next to it is reachable", None, None
            return self._walk(me, path) + [(adj[spot], None)], None, None, None
        return None, f"unknown intent kind {intent.kind!r}", None, None

    @staticmethod
    def _adjacent_floor(state, target) -> dict:
        """{floor tile next to target: action that bumps target from there}."""
        adj = {}
        for a, (dx, dy) in SCREEN_DELTAS.items():
            spot = (target[0] - dx, target[1] - dy)
            if _is_floor(state, *spot):
                adj[spot] = a
        return adj

    @staticmethod
    def _walk(start, path):
        steps, (x, y) = [], start
        for a in path:
            dx, dy = SCREEN_DELTAS[a]
            x, y = x + dx, y + dy
            steps.append((a, (x, y)))
        return steps

    def execute(self, intent: Intent | str, state: dict,
                step_fn: Callable[[int], VisibleStep | None]) -> ExecResult:
        """Run ``intent`` from ``state``. ``step_fn(action)`` performs one env step
        and returns a VisibleStep, or None when a budget is exhausted."""
        if isinstance(intent, str):
            intent = parse_intent(intent)
        res = ExecResult(intent=intent.text, kind=intent.kind)
        steps, err, note, res.coerced = self.plan(intent, state)
        if err is not None:
            res.error = f"{intent.text} failed: {err}"
            steps = [(WAIT, None)]  # budgets always advance
        elif not steps:
            res.note = f"{note}; waited one step" if note else "waited one step"
            steps = [(WAIT, None)]
        res.note = res.note or note

        stop = None
        for i, (action, expected) in enumerate(steps):
            if i >= self.max_macro_len:
                stop = "length_cap"
                break
            vs = step_fn(action)
            if vs is None:
                stop = "budget"
                break
            res.primitive_actions.append(int(action))
            res.events.append(vs.event)
            res.event_records.append(vs.state.get("last_event"))
            res.final = vs
            if vs.done:
                stop = "episode_end"
                break
            if expected is not None:
                ev = vs.state.get("last_event") or {}
                if ev.get("type") != "move" or tuple(vs.state["agent"]["pos"]) != tuple(expected):
                    stop = "unexpected_event"
                    break
        res.stop_reason = stop or ("invalid" if res.error else "completed")
        return res
