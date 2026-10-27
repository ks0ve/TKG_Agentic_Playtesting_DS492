"""Coordinate-free abstraction of agent-visible transitions (the TKG's expectation keys).

Built only from structured agent-visible data: ``env_adapter.visible_state`` dicts and the
event records (``held_before``, ``held_after``, ``tile``, ``tile_before``, ``tile_after``). No
observation text is parsed, so translator wording changes do not affect the keys.

Entity state strings (kind + visible state, never coordinates):
    plate:clean:[chopped tomato]      cutting board:holding:unchopped onion
    counter:empty                     floor          hands:nothing

ctx0    = (intent kind, hands before, target entity before)      intent kind: move | interact | wait
ctx1    = (previous outcome in this episode or START, ctx0)
outcome = (event type, success, hands after, target entity after)
coarse  = (intent kind, target kind)        target kind = the entity kind without its state
outcome_coarse = (event type, success)      what coarse-level expectations count

The target is the tile the intent's last step bumped or entered (its event record).
"""
from __future__ import annotations

START = "START"
_SEP = " | "
_KIND = {"goto": "move", "move": "move", "interact": "interact", "wait": "wait", "invalid": "wait"}


def item_str(rec: dict | None) -> str:
    """'unchopped onion', 'chopped cooked steak', 'plate:clean:[chopped tomato]', 'nothing'."""
    if rec is None:
        return "nothing"
    if rec.get("name") == "plate":
        contents = ", ".join(item_str(c) for c in rec.get("contents") or [])
        return f"plate:{'dirty' if rec.get('dirty') else 'clean'}:[{contents}]"
    states = []
    if "chopped" in rec:
        states.append("chopped" if rec["chopped"] else "unchopped")
    if "cooked" in rec:
        states.append("cooked" if rec["cooked"] else "uncooked")
    if rec.get("burned"):
        states.append("burned")
    return " ".join(states + [rec.get("name", "item")])


def hands_str(rec: dict | None) -> str:
    return f"hands:{item_str(rec)}"


_BARE = {"floor", "wall", "edge of the map", "another chef"}


def tile_str(tile: dict | None) -> str:
    """Entity string of an event record's ``tile_before`` / ``tile_after``."""
    if tile is None:
        return "none"
    kind = tile.get("kind") or "none"
    if kind in _BARE:
        return kind
    held = tile.get("holding")
    return f"{kind}:empty" if held is None else f"{kind}:holding:{item_str(held)}"


def tile_from_state(state: dict, x: int, y: int) -> str:
    """The same entity string as ``tile_str``, predicted from a visible state for tile (x, y)."""
    for st in state.get("stations", []):
        if tuple(st["pos"]) == (x, y):
            return tile_str({"kind": st["label"], "holding": st.get("holding")})
    for it in state.get("items", []):
        if it.get("where") == "counter" and tuple(it["pos"]) == (x, y):
            rec = {k: v for k, v in it.items() if k not in ("pos", "where")}
            return tile_str({"kind": "counter", "holding": rec})
    rows = state["map"]
    if not (0 <= y < len(rows) and 0 <= x < len(rows[y])):
        return "edge of the map"
    ch = rows[y][x]
    if ch in ".@":
        return "floor"
    if ch == "B":
        return "wall"
    return "counter:empty"


def intent_kind(kind: str) -> str:
    return _KIND.get(kind, "wait")


def ctx0_key(kind: str, held_before: dict | None, target_before: str) -> str:
    return _SEP.join((intent_kind(kind), hands_str(held_before), target_before))


def ctx1_key(prev_outcome: str | None, ctx0: str) -> str:
    return f"{prev_outcome or START} || {ctx0}"


def outcome_key(event: dict | None) -> str:
    ev = event or {}
    success = ev.get("success")
    return _SEP.join((str(ev.get("type") or "none"), "-" if success is None else str(success),
                      hands_str(ev.get("held_after")), tile_str(ev.get("tile_after"))))


def target_kind(entity: str) -> str:
    """'cutting board:holding:unchopped onion' -> 'cutting board'; 'floor' -> 'floor'."""
    return entity.split(":", 1)[0]


def coarse_key(kind: str, target: str) -> str:
    return _SEP.join((intent_kind(kind), target_kind(target)))


def outcome_coarse_key(event: dict | None) -> str:
    ev = event or {}
    success = ev.get("success")
    return _SEP.join((str(ev.get("type") or "none"), "-" if success is None else str(success)))


def abstract(kind: str, last_event: dict | None, prev_outcome: str | None) -> dict:
    """ctx0 / ctx1 / coarse / outcome / outcome_coarse / target of one executed intent (from its
    last event record)."""
    ev = last_event or {}
    target = tile_str(ev.get("tile_before"))
    c0 = ctx0_key(kind, ev.get("held_before"), target)
    return {"ctx0": c0, "ctx1": ctx1_key(prev_outcome, c0), "coarse": coarse_key(kind, target),
            "outcome": outcome_key(ev), "outcome_coarse": outcome_coarse_key(ev), "target": target}


def predicted_ctx0(intent, state: dict) -> str:
    """The ctx0 an intent would have if run from ``state`` (cheap-policy novelty)."""
    held = state["agent"]["holding"]
    if intent.kind in ("wait", "invalid"):
        return ctx0_key("wait", held, "none")
    if intent.kind == "goto":
        return ctx0_key("move", held, "floor")
    if intent.kind == "interact" and intent.x is not None:
        return ctx0_key("interact", held, tile_from_state(state, intent.x, intent.y))
    if intent.kind == "move":
        from react_baseline.action_reader import SCREEN_DELTAS
        dx, dy = SCREEN_DELTAS[intent.action]
        x, y = state["agent"]["pos"]
        target = tile_from_state(state, x + dx, y + dy)
        return ctx0_key("move" if target == "floor" else "interact", held, target)
    return ctx0_key(intent.kind, held, "none")


# -- readable forms for the LLM's MEMORY block (agent-visible strings only) ------------------
_VERBS = {"move": "moved", "none": "nothing happened", "pickup": "picked something up",
          "putdown": "put something down", "pickup_into_plate": "added something to the held plate",
          "chop": "chopped", "wash": "scrubbed a plate", "deliver": "delivered", "dump": "threw something away"}
_FAILED = {"putdown": "tried to put something down: refused",
           "pickup_into_plate": "tried to add something to the held plate: refused",
           "chop": "tried to chop: refused", "pickup": "tried to pick something up: refused"}


def entity_text(entity: str) -> str:
    """'cutting board:holding:unchopped onion' -> 'cutting board holding unchopped onion'."""
    if entity.startswith("hands:"):
        return entity[len("hands:"):]
    text = entity.replace(":holding:", " holding ").replace(":empty", " (empty)")
    for state in ("clean", "dirty"):
        text = text.replace(f"plate:{state}:[]", f"{state} empty plate").replace(f"plate:{state}:[",
                                                                                  f"{state} plate [")
    return text


def outcome_text(outcome: str) -> str:
    et, success, hands, target = outcome.split(_SEP)
    verb = _VERBS.get(et, "something happened")
    if et == "deliver":
        verb = "delivered: order ACCEPTED" if success == "True" else "delivered: order REJECTED"
    elif success == "False":
        verb = _FAILED.get(et, "tried something: refused")
    tail = f"; then holding {entity_text(hands)}"
    if target != "none":
        tail += f"; target now: {entity_text(target)}"
    return verb + tail


def outcome_coarse_text(outcome_coarse: str) -> str:
    et, success = outcome_coarse.split(_SEP)
    if et == "deliver":
        return "delivered: order ACCEPTED" if success == "True" else "delivered: order REJECTED"
    if success == "False":
        return _FAILED.get(et, "tried something: refused")
    return _VERBS.get(et, "something happened")


def coarse_text(coarse: str) -> str:
    kind, tkind = coarse.split(_SEP)
    return f"{kind} with a {tkind}" if tkind != "none" else kind


def situation_text(ctx0: str) -> str:
    kind, hands, target = ctx0.split(_SEP)
    return f"{kind}, holding {entity_text(hands)}; target {entity_text(target)}"
