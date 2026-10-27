"""The agent must never see oracle bug information."""
import json

from react_baseline import prompts
from react_baseline.agents import ReActAgent
from react_baseline.env_adapter import EnvAdapter  # sets up sys.path for overcookedPlus
from overcookedPlus.module.bug_manager import BugManager  # noqa: E402
from react_baseline.llm import MockLLM
from react_baseline.runner import Budgets, run

BUG_NAMES = [n for n in dir(BugManager) if n.startswith("BUG_")]
SENTINEL_EMPTY = 987654
SENTINEL_DELIVER = 987655
SENTINEL_COUNTER = 987656

# bump an empty counter; fetch the plate and deliver it empty; pick up the
# returned (dirty) plate and put it down on a counter
SCRIPT_ACTIONS = [3, 0, 0, 3, 1, 2, 3, 3, 2, 3]


def _leaks(text: str) -> list[str]:
    found = []
    for name in BUG_NAMES:
        if name in text or name[4:].lower() in text.lower():
            found.append(name)
    for s in (SENTINEL_EMPTY, SENTINEL_DELIVER, SENTINEL_COUNTER):
        if str(s) in text:
            found.append(str(s))
    for marker in ("bug_id", "bug id", "bug_manager", "info[", "predicate"):
        if marker in text.lower():
            found.append(marker)
    return found


def test_render_text_has_no_bug_info():
    adapter = EnvAdapter("V2_task_01_tomato")
    try:
        texts = [adapter.reset()]
        for a in SCRIPT_ACTIONS:
            res = adapter.step(a)
            texts += [res.obs_text, res.summary]
        for t in texts:
            assert "bug" not in t.lower(), t
            assert _leaks(t) == []
    finally:
        adapter.close()


def test_system_prompt_names_no_predicates():
    for k in (1, 3):
        assert _leaks(prompts.system_prompt(k)) == []


def _install_sentinels(monkeypatch):
    """Make predicates emit sentinel IDs so any leak of oracle output is detectable."""
    monkeypatch.setattr(BugManager, "check_pickup_from_empty", lambda self: [SENTINEL_EMPTY])
    orig_extra = BugManager.check_deliver_extra
    monkeypatch.setattr(BugManager, "check_deliver_extra",
                        lambda self, plate: orig_extra(self, plate) + [SENTINEL_DELIVER])
    orig_putdown = BugManager.check_putdown
    monkeypatch.setattr(BugManager, "check_putdown",
                        lambda self, *a, **k: orig_putdown(self, *a, **k) + [SENTINEL_COUNTER])


def _assert_sentinels_fired(run_dir):
    oracle = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
    seen = {b for r in oracle for b in r["bug_ids_triggered"]}
    assert {SENTINEL_EMPTY, SENTINEL_DELIVER, SENTINEL_COUNTER} <= seen  # oracle did fire


def _assert_clean(text: str):
    assert _leaks(text) == [], text[:500]
    assert "bug" not in text.lower(), text[:500]


# bump an empty counter, fetch the plate, deliver it empty, pick up the returned
# plate and put it on a counter: the same interactions as SCRIPT_ACTIONS, as intents
SCRIPT_INTENTS = ["interact(1,0)", "interact(3,0)", "interact(2,2)", "interact(plate)",
                  "interact(2,0)", "goto(9,9)", "wait(2)"]


class _ScriptedIntentAgent:
    name = "scripted-intent"
    action_space = "intent"

    def __init__(self, intents):
        self.intents = list(intents)
        self.feedback: list[str] = []   # executor traces + summaries the agent receives
        self.states: list[dict] = []    # structured states the agent receives

    def act(self, obs_text, state=None):
        from react_baseline.agents import Decision
        self.states.append(state)
        self.feedback.append(obs_text)
        return Decision(intents=[self.intents.pop(0) if self.intents else "wait"])

    def observe(self, executed, summary, results=None):
        self.feedback.append(summary)
        self.feedback += [r.trace for r in results or []]
        self.feedback += [json.dumps(r.to_dict()) for r in results or []]

    def on_episode_reset(self):
        pass


def test_structured_state_feedback_and_transitions_have_no_oracle(tmp_path, monkeypatch):
    _install_sentinels(monkeypatch)
    agent = _ScriptedIntentAgent(SCRIPT_INTENTS)
    run(agent, "V2_task_01_tomato", 0, Budgets(max_env_steps=40), tmp_path,
        {"backend": None, "model": None}, ignore_bug_ids=())
    _assert_sentinels_fired(tmp_path)
    assert any("goto(9,9) failed" in f for f in agent.feedback)  # executor errors reach the agent
    for f in agent.feedback:
        _assert_clean(f)
    for s in agent.states:
        _assert_clean(json.dumps(s))
    text = (tmp_path / "transitions.jsonl").read_text()
    assert text.count("\n") >= len(SCRIPT_INTENTS)
    _assert_clean(text)


def test_intent_react_prompts_never_contain_oracle(tmp_path, monkeypatch):
    """Intent-mode ReAct: prompts carry executor traces/errors and history; none of it
    may carry oracle output. Also covers the logged executor_result."""
    _install_sentinels(monkeypatch)
    script = [f"Thought: step {i}.\nAction: {it}" for i, it in enumerate(SCRIPT_INTENTS)]
    llm = MockLLM(script=script)
    agent = ReActAgent(llm, history=50, action_space="intent")
    run(agent, "V2_task_01_tomato", 0, Budgets(max_env_steps=40, max_llm_calls=len(script)),
        tmp_path, {"backend": "mock", "model": "mock"}, ignore_bug_ids=())
    _assert_sentinels_fired(tmp_path)
    assert any("Executor: goto(9,9)" in u for _, u in llm.calls)
    for system, user in llm.calls:
        assert _leaks(system) == []
        assert _leaks(user) == [], user
    for row in (tmp_path / "llm_calls.jsonl").read_text().splitlines():
        rec = json.loads(row)
        _assert_clean(json.dumps(rec["executor_result"]))
        _assert_clean(json.dumps(rec["parsed_intents"]))
    for p in (prompts.system_prompt(1, "intent"), prompts.system_prompt(3, "intent")):
        assert _leaks(p) == []


def test_agent_prompts_never_contain_oracle(tmp_path, monkeypatch):
    _install_sentinels(monkeypatch)

    script = [f"Thought: step {i}.\nAction: {a}" for i, a in enumerate(SCRIPT_ACTIONS)]
    llm = MockLLM(script=script)
    agent = ReActAgent(llm, history=50)
    run(agent, "V2_task_01_tomato", 0, Budgets(max_env_steps=len(SCRIPT_ACTIONS)),
        tmp_path, {"backend": "mock", "model": "mock"}, ignore_bug_ids=())

    oracle = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    seen = {b for r in oracle for b in r["bug_ids_triggered"]}
    assert {SENTINEL_EMPTY, SENTINEL_DELIVER, SENTINEL_COUNTER} <= seen  # oracle did fire
    assert len(llm.calls) == len(SCRIPT_ACTIONS)
    for system, user in llm.calls:
        assert _leaks(system) == []
        assert _leaks(user) == [], user
    # the logged prompts are the same ones the model saw
    logged = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    assert [r["prompt"] for r in logged] == [u for _, u in llm.calls]


def test_tkg_memory_block_has_no_oracle_data(tmp_path, monkeypatch):
    """The TKG arm's MEMORY block ("The automatic explorer (not you) just did: <intent> -> <event>."
    + the fixed investigation instruction) is built from the TKG only: no oracle data reaches it."""
    from memory.trigger import EveryNTrigger
    from react_baseline.tkg_agent import INVESTIGATE, TKGAgent
    _install_sentinels(monkeypatch)
    llm = MockLLM(script=["Thought: a\nAction: interact(1,0)", "Thought: b\nAction: interact(3,0)",
                          "Thought: c\nAction: interact(2,2)", "Thought: d\nAction: resume"])
    run(TKGAgent(llm, trigger=EveryNTrigger(2)), "V2_task_01_tomato", 0, Budgets(max_env_steps=150), tmp_path,
        {"backend": "mock", "model": "mock"})
    oracle = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    seen = {b for r in oracle for b in r["bug_ids_triggered"]}
    assert {SENTINEL_EMPTY, SENTINEL_COUNTER} & seen     # the oracle did emit sentinels this run
    blocks = [u[u.index("MEMORY:"):u.index(INVESTIGATE) + len(INVESTIGATE)] for _, u in llm.calls if "MEMORY:" in u]
    assert blocks and any("The automatic explorer (not you) just did:" in b for b in blocks)
    for b in blocks:
        head = b.split("\n", 1)[0]
        _assert_clean(head)                       # no bug names, sentinels, oracle markers, or "bug"
        assert _leaks(b) == []                    # the instruction's own wording is oracle-free too
    for _, user in llm.calls:
        assert _leaks(user) == []
