"""TKG agent: warm-up without LLM, escalation for <= k decisions with resume, MEMORY block and
recorded path free of oracle data, repro path only when enabled, mock end-to-end."""
import json
import zipfile

import pytest

from memory.trigger import EveryNTrigger, Trigger
from react_baseline import bundle, run as run_cli
from react_baseline.llm import MockIntentLLM, MockLLM
from react_baseline.repro import ReproConfig
from react_baseline.runner import Budgets, run
from react_baseline.tests.test_display_names import _raw_hits
from react_baseline.tests.test_prompt_leak import _assert_sentinels_fired, _install_sentinels, _leaks
from react_baseline.tkg_agent import TKGAgent

CFG = {"backend": "mock", "model": "mock"}
TASK = "V2_task_01_tomato"


class OnceTrigger(Trigger):
    """Fires on the first check only."""
    name = "once"

    def __init__(self):
        super().__init__()
        self.n = 0

    def check(self, transition, tkg):
        self.detail = self._memory_detail(transition, tkg, transition["ctx1"])
        self.n += 1
        return (True, "novel") if self.n == 1 else (False, None)


def _rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_no_llm_calls_during_warmup(tmp_path):
    llm = MockIntentLLM(seed=0)
    s = run(TKGAgent(llm, warmup_steps=1000), TASK, 0, Budgets(max_env_steps=300), tmp_path / "a", CFG)
    assert s["llm_calls"] == 0 and llm.calls == [] and s["trigger_checks"] == 0 and s["escalations"] == 0
    llm = MockIntentLLM(seed=0)
    s = run(TKGAgent(llm, warmup_steps=100, trigger=EveryNTrigger(5)), TASK, 0, Budgets(max_env_steps=300),
            tmp_path / "b", CFG)
    calls = _rows(tmp_path / "b" / "llm_calls.jsonl")
    assert calls and min(c["global_step"] for c in calls) >= 100
    trans = _rows(tmp_path / "b" / "transitions.jsonl")
    assert all(t["deviation"] is None for t in trans if t["global_step_start"] <= 100)


@pytest.mark.parametrize("reply,n_llm", [("Thought: look\nAction: wait", 3), ("Thought: fine\nAction: resume", 1)])
def test_escalation_gives_llm_at_most_k_decisions_and_resume(tmp_path, reply, n_llm):
    llm = MockLLM(script=[reply])
    agent = TKGAgent(llm, warmup_steps=20, investigate_k=3, trigger=OnceTrigger())
    s = run(agent, TASK, 0, Budgets(max_env_steps=150), tmp_path, CFG)
    calls = _rows(tmp_path / "llm_calls.jsonl")
    assert len(calls) == n_llm == s["llm_calls"] and s["escalations"] == 1
    assert [c["trigger_reason"] for c in calls] == ["novel", "investigate", "investigate"][:n_llm]
    decisions = sorted({c["decision_id"] for c in calls})
    assert decisions == list(range(decisions[0], decisions[0] + n_llm))   # consecutive, then cheap again
    trans = _rows(tmp_path / "transitions.jsonl")
    llm_trans = [t for t in trans if t["actor"] == "llm"]
    if reply.endswith("resume"):
        assert calls[0]["resumed"] is True and calls[0]["parse_error"] is None and not llm_trans
        assert s["llm_resumes"] == 1
    else:
        assert len(llm_trans) == 3 and all(t["intent"] == "wait" for t in llm_trans)
    later = [t for t in trans if t["decision_id"] > decisions[-1]]
    assert later and all(t["actor"] == "cheap" for t in later)
    assert "[auto] [turn" in llm.calls[0][1]             # cheap turns are shown as [auto] in the history
    assert "MEMORY: " in llm.calls[0][1] and 'Action: resume' in llm.calls[0][1]


def test_memory_block_and_recorded_path_have_no_oracle_data(tmp_path, monkeypatch):
    _install_sentinels(monkeypatch)
    script = ["Thought: t\nAction: interact(1,0)", "Thought: u\nAction: interact(3,0)",
              "Thought: v\nAction: interact(2,2)\nBug report: the plate came back strangely",
              "Thought: w\nAction: wait"]
    llm = MockLLM(script=script)
    agent = TKGAgent(llm, warmup_steps=30, trigger=EveryNTrigger(1), investigate_k=5)
    blocks = []
    real = agent.react.build_prompt

    def spy(obs):
        if agent.react.extra_block:
            blocks.append(agent.react.extra_block)
        return real(obs)
    monkeypatch.setattr(agent.react, "build_prompt", spy)
    run(agent, TASK, 0, Budgets(max_env_steps=400), tmp_path, CFG,
        repro=ReproConfig(enabled=True, steps=20, max_fraction=0.5))
    _assert_sentinels_fired(tmp_path)
    mem = [b for b in blocks if b.startswith("MEMORY:")]
    path = [b for b in blocks if b.startswith("RECORDED PATH")]
    assert mem and path
    for b in mem + path:
        assert _leaks(b) == [] and "bug" not in b.lower(), b[:300]
        assert not _raw_hits(b), (_raw_hits(b), b[:300])
    for _, user in llm.calls:
        assert _leaks(user) == [], user[-400:]
    graph = (tmp_path / "tkg.json").read_text(encoding="utf-8")
    assert _leaks(graph) == [] and "bug" not in graph.lower()


@pytest.mark.parametrize("repro_path", [True, False])
def test_repro_path_only_when_enabled(tmp_path, repro_path):
    script = ["Thought: a\nAction: interact(0,2)\nBug report: the tomato looked odd", "Thought: b\nAction: wait"]
    llm = MockLLM(script=script)
    agent = TKGAgent(llm, warmup_steps=10, trigger=EveryNTrigger(1), repro_path=repro_path)
    run(agent, TASK, 0, Budgets(max_env_steps=120), tmp_path, CFG,
        repro=ReproConfig(enabled=True, steps=10, max_fraction=0.5))
    calls = _rows(tmp_path / "llm_calls.jsonl")
    repro_calls = [c for c in calls if c["phase"] == "repro"]
    assert repro_calls and all(c["trigger_reason"] == "repro" for c in repro_calls)
    assert all(("RECORDED PATH" in c["prompt"]) is repro_path for c in repro_calls)
    assert not any("RECORDED PATH" in c["prompt"] for c in calls if c["phase"] == "explore")
    if repro_path:
        assert "interact(0,2) -> picked up tomato" in repro_calls[0]["prompt"]


def test_tkg_determinism(tmp_path):
    def go(d):
        run(TKGAgent(MockIntentLLM(seed=4), seed=4, warmup_steps=50), "V2_task_07_tomato_steak", 4,
            Budgets(max_env_steps=300), tmp_path / d, CFG)
        return (tmp_path / d / "tkg.json").read_text(), (tmp_path / d / "steps.jsonl").read_text()
    assert go("a") == go("b")


def test_mock_end_to_end_cli_and_bundle(tmp_path):
    rd = tmp_path / "run"
    assert run_cli.main(["--agent", "tkg", "--llm", "mock", "--task", "V2_task_11_tomato_lettuce_onion",
                         "--max-env-steps", "1000", "--repro", "--run-dir", str(rd), "--quiet"]) == 0
    for f in ("config.json", "summary.json", "steps.jsonl", "llm_calls.jsonl", "transitions.jsonl",
              "episodes.jsonl", "bug_reports.jsonl", "repro_attempts.jsonl", "tkg.json"):
        assert (rd / f).exists(), f
    s = json.loads((rd / "summary.json").read_text())
    for k in ("escalations", "llm_calls_per_env_step", "warmup_steps", "tkg_nodes", "tkg_edges",
              "trigger_fire_rate", "deviation_reasons"):
        assert k in s, k
    assert s["total_env_steps"] == 1000 and s["warmup_steps"] == 200 and s["trigger"] == "tkg1"
    calls = _rows(rd / "llm_calls.jsonl")
    assert calls and all(c["trigger_reason"] in ("novel", "surprising", "investigate", "repro") for c in calls)
    trans = _rows(rd / "transitions.jsonl")
    assert {t["actor"] for t in trans} == {"cheap", "llm"}
    out = tmp_path / "b.zip"
    assert bundle.main([str(rd), "-o", str(out)]) == 0
    assert any(n.endswith("/tkg.json") for n in zipfile.ZipFile(out).namelist())
    assert run_cli.main(["--agent", "cheap", "--max-env-steps", "50", "--run-dir", str(tmp_path / "c"),
                         "--quiet"]) == 0
    assert (tmp_path / "c" / "tkg.json").exists()


def test_tkg1c_memory_blocks_are_clean(tmp_path, monkeypatch):
    from memory.trigger import make_trigger
    _install_sentinels(monkeypatch)
    llm = MockIntentLLM(seed=2)
    agent = TKGAgent(llm, warmup_steps=0, trigger=make_trigger("tkg1c"))
    run(agent, "V2_task_11_tomato_lettuce_onion", 0, Budgets(max_env_steps=400), tmp_path, CFG)
    blocks = [u[u.index("MEMORY:"):].split("\n\n")[0] for _, u in llm.calls if "MEMORY:" in u]
    assert blocks and any("In similar situations" in b for b in blocks)
    for b in blocks:
        assert _leaks(b) == [] and "bug" not in b.lower() and not _raw_hits(b), b


def test_consecutive_llm_cap_across_restarts(tmp_path):
    # every check fires, so each LLM transition restarts k = 3; the cap of 6 still hands control back
    llm = MockLLM(script=["Thought: look\nAction: wait"])
    agent = TKGAgent(llm, warmup_steps=0, investigate_k=3, trigger=EveryNTrigger(1), max_consecutive_llm=6)
    s = run(agent, TASK, 0, Budgets(max_env_steps=60), tmp_path, CFG)
    trans = _rows(tmp_path / "transitions.jsonl")
    runs, cur = [], 0
    for t in trans:
        if t["actor"] == "llm":
            cur += 1
        else:
            runs.append(cur)
            cur = 0
    assert max(runs + [cur]) == 6
    released = [t for t in trans if t["trigger_reason"] == "cap_release"]
    assert released and all(t["actor"] == "cheap" for t in released) and s["cap_releases"] == len(released)
    assert {t["trigger_reason"] for t in trans if t["actor"] == "llm"} <= {"every_n", "investigate"}
    assert s["max_consecutive_llm"] == 6
