"""TKG store: counts, backoff, NEXT chaining across episodes, JSON export matching the schema."""
import json

from memory.abstraction import ctx1_key
from memory.tkg import SCHEMA, TKG, state_entities, state_id

C0 = "interact | hands:unchopped tomato | cutting board:empty"
PUT = "putdown | True | hands:nothing | cutting board:holding:unchopped tomato"
FAIL = "putdown | False | hands:unchopped tomato | cutting board:empty"


def _state(held=None, step=0, board=None):
    return {"episode_step": step, "map": ["#K#", ".@."], "agent": {"pos": [1, 1], "holding": held},
            "items": [{"name": "tomato", "chopped": False, "pos": [0, 0], "where": "counter"}],
            "stations": [{"kind": "knife", "label": "cutting board", "pos": [1, 0], "holding": board,
                          "progress": None}],
            "order": [["tomato"]], "last_event": {"text": f"step {step}"}}


def _t(tkg, i, episode, outcome=PUT, prev=None, ctx0=C0):
    ek = f"run#ep{episode}"
    prev = tkg.prev_outcome(ek) if prev is None else prev
    t = {"id": f"t{i}", "run": "run", "episode": episode, "episode_key": ek, "global_step_start": i,
         "global_step_end": i, "intent": "interact(1,0)", "ctx0": ctx0, "ctx1": ctx1_key(prev, ctx0),
         "outcome": outcome, "target": ctx0.split(" | ")[2], "actor": "cheap", "deviation": None,
         "escalated": False}
    return tkg.add(t, _state(step=i), _state(step=i + 1))


def test_state_id_ignores_step_and_event_and_entities_dedupe():
    assert state_id(_state(step=1)) == state_id(_state(step=9))
    assert state_id(_state()) != state_id(_state(held={"name": "tomato", "chopped": False}))
    assert state_entities(_state()) == ["counter:holding:unchopped tomato", "cutting board:empty",
                                        "hands:nothing"]


def test_counts_and_backoff():
    tkg = TKG(min_support=3)
    assert tkg.count(C0) == 0 and tkg.expect("x", C0) == ({}, "ctx0", 0, f"exp:ctx0:{C0}")
    for i in range(3):
        _t(tkg, i, episode=i + 1)            # each first in its episode: ctx1 = START || C0
    _t(tkg, 10, episode=1, outcome=FAIL)     # after PUT in episode 1
    assert tkg.count(C0) == 4
    dist, level, support, eid = tkg.expect(ctx1_key(None, C0), C0)
    assert (level, support, dist) == ("ctx1", 3, {PUT: 1.0}) and eid == f"exp:ctx1:START || {C0}"
    dist, level, support, _ = tkg.expect(ctx1_key(PUT, C0), C0)      # ctx1 seen once -> back off
    assert level == "ctx0" and support == 4 and dist == {PUT: 0.75, FAIL: 0.25}
    assert tkg.expect(None, C0)[1] == "ctx0"                          # order 0: ctx0 only
    e = tkg.expectations[f"exp:ctx0:{C0}"]
    assert (e.first_seen_step, e.last_seen_step, e.source) == (0, 10, "observed")


def test_next_chains_within_episodes_only():
    tkg = TKG()
    _t(tkg, 1, 1)
    _t(tkg, 2, 1)
    _t(tkg, 3, 2)      # new episode (reset): no NEXT from t2
    _t(tkg, 4, 1)      # back in episode 1 (e.g. a later transition of the same episode key)
    _t(tkg, 5, 2)
    nxt = [(e["source"], e["target"]) for e in tkg.edges if e["type"] == "NEXT"]
    assert nxt == [("t1", "t2"), ("t2", "t4"), ("t3", "t5")]
    assert [t["id"] for t in tkg.episode_path("run#ep1", upto="t2")] == ["t1", "t2"]
    assert [t["id"] for t in tkg.episode_path("run#ep2")] == ["t3", "t5"]
    assert tkg.transitions["t3"]["ctx1"].startswith("START || ")


def test_export_matches_schema(tmp_path):
    tkg = TKG()
    _t(tkg, 1, 1)
    tkg.add({**tkg.transitions["t1"], "id": "t2", "global_step_end": 2}, _state(), _state(), "exp:ctx0:" + C0)
    tkg.export_json(tmp_path / "tkg.json")
    data = json.loads((tmp_path / "tkg.json").read_text())
    types = {n["type"] for n in data["nodes"]}
    assert types == set(SCHEMA["nodes"])
    assert {e["type"] for e in data["edges"]} == set(SCHEMA["edges"])
    ids = {n["id"] for n in data["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in data["edges"])
    allowed = {"NEXT": ("Transition", "Transition"), "FROM": ("Transition", "State"),
               "TO": ("Transition", "State"), "TARGET": ("Transition", "Entity"),
               "ASSESSED_AGAINST": ("Transition", "Expectation"), "HAS_ENTITY_STATE": ("State", "Entity")}
    by_id = {n["id"]: n["type"] for n in data["nodes"]}
    for e in data["edges"]:
        assert (by_id[e["source"]], by_id[e["target"]]) == allowed[e["type"]], e
    assert data["n_nodes"] == tkg.n_nodes and data["n_edges"] == len(data["edges"])
