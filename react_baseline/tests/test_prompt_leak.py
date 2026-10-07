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


def test_agent_prompts_never_contain_oracle(tmp_path, monkeypatch):
    # Make predicates emit sentinel IDs so any leak of oracle output is detectable.
    monkeypatch.setattr(BugManager, "check_pickup_from_empty", lambda self: [SENTINEL_EMPTY])
    orig_extra = BugManager.check_deliver_extra
    monkeypatch.setattr(BugManager, "check_deliver_extra",
                        lambda self, plate: orig_extra(self, plate) + [SENTINEL_DELIVER])
    orig_putdown = BugManager.check_putdown
    monkeypatch.setattr(BugManager, "check_putdown",
                        lambda self, *a, **k: orig_putdown(self, *a, **k) + [SENTINEL_COUNTER])

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
