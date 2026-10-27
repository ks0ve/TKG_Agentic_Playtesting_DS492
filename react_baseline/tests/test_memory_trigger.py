"""Memory check: novel vs surprising vs expected on a handcrafted sequence; random / every-n rates."""
from memory.abstraction import ctx1_key
from memory.tkg import TKG
from memory.trigger import EveryNTrigger, RandomTrigger, TKGTrigger, make_trigger

C0 = "interact | hands:unchopped tomato | cutting board:empty"
OTHER = "interact | hands:nothing | sink:empty"
PUT = "putdown | True | hands:nothing | cutting board:holding:unchopped tomato"
FAIL = "putdown | False | hands:unchopped tomato | cutting board:empty"
STATE = {"episode_step": 0, "map": ["."], "agent": {"pos": [0, 0], "holding": None}, "items": [],
         "stations": [], "order": [], "last_event": None}
_n = [0]


def _tr(ctx0, outcome, prev=None):
    _n[0] += 1
    return {"id": f"t{_n[0]}", "episode_key": f"e{_n[0]}", "global_step_start": _n[0], "global_step_end": _n[0],
            "ctx0": ctx0, "ctx1": ctx1_key(prev, ctx0), "outcome": outcome, "target": ctx0.split(" | ")[2]}


def _check_then_add(trigger, tkg, t):
    fire = trigger.check(t, tkg)
    tkg.add(t, STATE, STATE)
    return fire


def test_novel_surprising_expected():
    tkg, trig = TKG(), TKGTrigger(order=1, min_support=3, tau=0.15)
    assert _check_then_add(trig, tkg, _tr(C0, PUT)) == (True, "novel")
    assert _check_then_add(trig, tkg, _tr(C0, PUT)) == (False, None)    # support 1 < 3: not judged yet
    for _ in range(4):
        _check_then_add(trig, tkg, _tr(C0, PUT))
    assert trig.check(_tr(C0, PUT), tkg) == (False, None)                # expected outcome
    assert trig.detail["level"] == "ctx1" and trig.detail["usual"] == PUT and trig.detail["usual_count"] == 6
    assert trig.check(_tr(C0, FAIL), tkg) == (True, "surprising")        # P = 0 < tau at support 6
    assert trig.detail["p_outcome"] == 0.0
    assert trig.check(_tr(OTHER, PUT), tkg) == (True, "novel")


def test_order1_uses_temporal_context_order0_does_not():
    tkg = TKG()
    # after a FAIL, putting down again fails (6x); from START it succeeds (6x)
    for _ in range(6):
        tkg.add(_tr(C0, PUT), STATE, STATE)
        tkg.add(_tr(C0, FAIL, prev=FAIL), STATE, STATE)
    t = _tr(C0, FAIL, prev=FAIL)
    assert TKGTrigger(order=1).check(t, tkg) == (False, None)     # expected after a FAIL
    t0 = _tr(C0, FAIL, prev=None)
    assert TKGTrigger(order=1).check(t0, tkg) == (True, "surprising")  # never fails from START
    assert TKGTrigger(order=0).check(t0, tkg) == (False, None)    # 50% under ctx0 alone


def test_random_and_every_n_rates():
    tkg = TKG()
    r = RandomTrigger(p=0.25, seed=3)
    fires = sum(r.check(_tr(C0, PUT), tkg)[0] for _ in range(4000))
    assert 0.22 < fires / 4000 < 0.28
    e = EveryNTrigger(n=7)
    hits = [i for i in range(70) if e.check(_tr(C0, PUT), tkg)[0]]
    assert len(hits) == 10 and hits[0] == 6 and e.check(_tr(C0, PUT), tkg)[1] is None
    assert make_trigger("tkg0").order == 0 and make_trigger("every_n", n=4).n == 4
    assert make_trigger("random", p=0.5).p == 0.5 and make_trigger("tkg1", tau=0.3).tau == 0.3


def _trc(ctx0, outcome, prev=None):
    """A transition with the coarse fields, as memory.abstraction.abstract builds them."""
    kind, _, target = ctx0.split(" | ")
    t = _tr(ctx0, outcome, prev)
    t["coarse"] = f"{kind} | {target.split(':', 1)[0]}"
    t["outcome_coarse"] = " | ".join(outcome.split(" | ")[:2])
    return t


ONION_C0 = "interact | hands:unchopped onion | cutting board:empty"
PUT_ONION = "putdown | True | hands:nothing | cutting board:holding:unchopped onion"
FAIL_ONION = "putdown | False | hands:unchopped onion | cutting board:empty"


def test_tkg1c_coarse_fallback():
    tkg, trig = TKG(), make_trigger("tkg1c")
    assert trig.name == "tkg1c" and trig.coarse
    assert _check_then_add(trig, tkg, _trc(C0, PUT)) == (True, "novel")       # coarse unseen
    for _ in range(3):
        _check_then_add(trig, tkg, _trc(C0, PUT))
    # a new ctx0 (onion instead of tomato) on a known coarse (interact | cutting board)
    assert trig.check(_trc(ONION_C0, PUT_ONION), tkg) == (False, None)       # "putdown | True" seen: expected
    assert trig.detail["level"] == "coarse" and trig.detail["usual_text"] == "put something down"
    assert trig.check(_trc(ONION_C0, FAIL_ONION), tkg) == (True, "surprising")  # refused: P = 0 at coarse
    assert trig.detail["level"] == "coarse"
    # tkg1 would call the same new ctx0 novel
    assert TKGTrigger(order=1).check(_trc(ONION_C0, PUT_ONION), tkg) == (True, "novel")
    # with ctx0 support >= min_support, tkg1c judges at ctx1/ctx0 like tkg1
    assert trig.check(_trc(C0, FAIL), tkg) == (True, "surprising") and trig.detail["level"] == "ctx1"
    assert trig.check(_trc(C0, PUT), tkg) == (False, None)
    assert tkg.expectations["exp:coarse:interact | cutting board"].outcomes == {"putdown | True": 4}
