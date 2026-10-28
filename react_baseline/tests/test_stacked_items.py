"""Translator v2.2: two items can share a counter tile (SAGE respawns a thrown-away or delivered item
on its start tile even when another item lies there: ActionWrapper.reset_item -> MovableItem.refresh
+ MapManager.place_item). Every item on a tile must be visible: ITEMS, ADJACENT, the legend note,
event records ("also") and visible_state (hidden / stacked flags)."""
from memory.abstraction import abstract, predicted_ctx0, tile_str
from react_baseline.action_reader import ActionReader, VisibleStep, parse_intent
from react_baseline.env_adapter import STACK_NOTE, EnvAdapter

# tomato map: steak spawns at (0,1), tomato at (0,2), counter (1,0) is empty, trash can at (1,4)
STACK = ["interact(0,1)",   # pick up the steak
         "interact(1,0)",   # put it on the empty counter (1,0)
         "interact(0,2)",   # pick up the tomato
         "interact(0,1)",   # put the tomato on the steak's (now empty) start tile
         "interact(1,0)",   # pick the steak up again
         "interact(1,4)"]   # throw it away: it respawns at (0,1), on top of the tomato


def _play(intents):
    a, reader, out = EnvAdapter("V2_task_01_tomato"), ActionReader(), []
    a.reset()
    state = a.state

    def step(act):
        r = a.step(act)
        out.append(r)
        return VisibleStep(r.obs_text, r.summary, r.state, r.state["last_event"]["text"], r.done)

    results = []
    for it in intents:
        intent = parse_intent(it)
        res = reader.execute(intent, state, step)
        results.append((intent, state, res))
        state = res.final.state
    return a, out, results


def test_respawn_stacks_two_items_and_all_are_shown():
    a, out, results = _play(STACK)
    try:
        obs, st = out[-1].obs_text, out[-1].state
        assert results[-1][2].event_records[-1]["type"] == "dump"
        items = [i for i in st["items"] if tuple(i["pos"]) == (0, 1)]
        assert sorted(i["name"] for i in items) == ["steak", "tomato"]          # two items, one tile
        assert all(i.get("stacked") for i in items)
        assert [i["name"] for i in items if i.get("hidden")] == ["tomato"]       # the map shows the steak
        assert st["map"][1][0] == "S" and STACK_NOTE in obs
        assert "- (0,1) steak [unchopped, uncooked] (on counter; also on this tile: tomato [unchopped])" in obs
        assert "- (0,1) tomato [unchopped] (on counter; the map shows steak [unchopped, uncooked] on this tile)" in obs
    finally:
        a.close()


def test_hidden_item_after_the_top_one_is_taken():
    a, out, results = _play(STACK + ["goto(1,1)", "interact(0,1)", "interact(1,0)", "interact(0,1)"])
    try:
        take_steak, put_away, bump = results[-3][2], results[-2][2], results[-1][2]
        ev = take_steak.event_records[-1]
        assert ev["type"] == "pickup" and ev["held_after"]["name"] == "steak"        # the game takes the map's item
        assert ev["tile_before"]["holding"]["name"] == "steak"
        assert [i["name"] for i in ev["tile_before"]["also"]] == ["tomato"]
        assert ev["tile_after"] == {"kind": "counter", "holding": None, "also": [{"name": "tomato", "chopped": False}]}
        # the tomato is still there but the map shows an empty counter: the game ignores it
        ev = bump.event_records[-1]
        assert ev["type"] is None and ev["held_after"] is None
        assert ev["text"] == "nothing happened (bumped empty counter (also on this tile: tomato) at (0,1))"
        obs = out[-1].obs_text
        assert "- 2 left  -> (0,1): counter shown empty; on this tile: tomato [unchopped]" in obs
        assert "- (0,1) tomato [unchopped] (on counter; the map shows an empty counter on this tile)" in obs
        assert tile_str(ev["tile_before"]) == "counter:empty+also:unchopped tomato"
        # the cheap policy's prediction sees the same tile
        intent, state, res = results[-1]
        assert predicted_ctx0(intent, state) == abstract(res.kind, ev, None)["ctx0"]
    finally:
        a.close()
