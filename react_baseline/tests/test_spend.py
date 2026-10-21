"""Paid-run guardrails: ledger, budget cap, reservations, CLI rules, Anthropic retry policy.
Offline only: the 'paid' backend is the mock backend with a nonzero price."""
import json
from types import SimpleNamespace

import httpx2
import pytest

from react_baseline import llm, run as run_cli, spend
from react_baseline.agents import ReActAgent
from react_baseline.llm import AnthropicLLM, MockLLM, PaidAPIError, resolve_profile
from react_baseline.runner import Budgets, run

PAID_YAML = """\
defaults: {mock: mock, ollama: free}
profiles:
  mock: {backend: mock, model: mock}
  free: {backend: ollama, model: x:1b}
  mock-paid:
    backend: mock
    model: mock-paid
    price_in_per_mtok: 1.0
    price_out_per_mtok: 5.0
  mock-off:
    backend: mock
    model: mock-off
    disabled: true
"""


@pytest.fixture
def paid_env(tmp_path, monkeypatch):
    y = tmp_path / "models.yaml"
    y.write_text(PAID_YAML, encoding="utf-8")
    monkeypatch.setattr(llm, "MODELS_YAML", y)
    b = tmp_path / "budget.yaml"
    b.write_text("project_cap_usd: 20.00\nsafety_margin_usd: 1.00\n", encoding="utf-8")
    led = tmp_path / "ledger.jsonl"
    return SimpleNamespace(models=y, budget=b, ledger=led, out=tmp_path / "runs")


def _ledger(env):
    return spend.Ledger(env.ledger, spend.load_budget(env.budget))


def _cli(env, *extra):
    return ["--agent", "react", "--model-profile", "mock-paid", "--max-env-steps", "10",
            "--out-dir", str(env.out), "--ledger", str(env.ledger), "--budget-file", str(env.budget),
            "--quiet", *extra]


# ---------------------------------------------------------------- ledger
def test_ledger_blocks_run_that_would_exceed_cap(paid_env):
    L = _ledger(paid_env)
    L.reserve(paid_env.out / "a", "mock-paid", "m", 2.0)
    L.finalize(paid_env.out / "a", 17.5, 1, 1, "completed")
    assert L.spent() == pytest.approx(17.5) and L.remaining() == pytest.approx(1.5)
    L.preflight(1.5)                        # exactly at cap - margin is allowed
    with pytest.raises(spend.SpendGuardError, match="Refusing to start a paid run"):
        L.preflight(1.51)


def test_stale_reservation_counts_full_until_reconciled(paid_env):
    L = _ledger(paid_env)
    run_dir = paid_env.out / "crashed"
    run_dir.mkdir(parents=True)
    L.reserve(run_dir, "mock-paid", "m", 2.0)
    assert L.committed() == pytest.approx(2.0) and len(L.open_reservations()) == 1
    (run_dir / "summary.json").write_text(json.dumps({"usd_cost": 0.31, "input_tokens": 10,
                                                      "output_tokens": 2, "stop_reason": "error: X"}))
    assert spend.main(["--ledger", str(paid_env.ledger), "--budget", str(paid_env.budget),
                       "--reconcile", str(run_dir)]) == 0
    assert L.open_reservations() == [] and L.spent() == pytest.approx(0.31)
    assert _ledger(paid_env).state()[1][spend._key(run_dir)]["status"] == "reconciled"


def test_run_writes_and_finalizes_reservation(paid_env):
    prof = resolve_profile("mock-paid")
    cfg = {"backend": "mock", "model": "mock-paid", "model_profile": prof.to_dict()}
    s = run(ReActAgent(MockLLM()), "V2_task_01_tomato", 0, Budgets(max_env_steps=5, max_usd=0.5),
            paid_env.out / "ok", cfg, ledger=_ledger(paid_env))
    recs = [json.loads(line) for line in paid_env.ledger.read_text().splitlines()]
    assert [r["type"] for r in recs] == ["reservation", "final"]
    assert recs[0]["status"] == "running" and recs[0]["max_usd"] == 0.5 and recs[0]["profile"] == "mock-paid"
    assert recs[1]["status"] == "completed" and recs[1]["usd_cost"] == pytest.approx(s["usd_cost"])
    assert s["usd_cost"] > 0


def test_reservation_finalized_on_exception_and_interrupt(paid_env):
    prof = resolve_profile("mock-paid").to_dict()
    cfg = {"backend": "mock", "model": "mock-paid", "model_profile": prof}
    with pytest.raises(Exception):   # context overflow inside the loop
        run(ReActAgent(MockLLM(context_limit=10)), "V2_task_01_tomato", 0,
            Budgets(max_env_steps=5, max_usd=0.5), paid_env.out / "err", cfg, ledger=_ledger(paid_env))

    class Interrupting(ReActAgent):
        def act(self, obs_text, state=None):
            if self._turn_index >= 2:
                raise KeyboardInterrupt
            return super().act(obs_text, state)
    run(Interrupting(MockLLM()), "V2_task_01_tomato", 0, Budgets(max_env_steps=50, max_usd=0.5),
        paid_env.out / "int", cfg, ledger=_ledger(paid_env))
    _, fin = _ledger(paid_env).state()
    statuses = {k.rsplit("/", 1)[-1]: v["status"] for k, v in fin.items()}
    assert statuses == {"err": "error", "int": "interrupted"}
    assert _ledger(paid_env).open_reservations() == []


# ---------------------------------------------------------------- CLI rules
def test_paid_profile_requires_allow_paid_and_max_usd(paid_env):
    with pytest.raises(SystemExit):
        run_cli.parse_args(_cli(paid_env, "--max-usd", "0.5"))            # no --allow-paid
    with pytest.raises(SystemExit):
        run_cli.parse_args(_cli(paid_env, "--allow-paid"))                # no --max-usd
    with pytest.raises(SystemExit):
        run_cli.parse_args(_cli(paid_env, "--allow-paid", "--max-usd", "3"))   # above $2 ceiling
    a = run_cli.parse_args(_cli(paid_env, "--allow-paid", "--max-usd", "3", "--max-usd-override"))
    assert a.max_usd == 3.0
    run_cli.parse_args(["--agent", "random", "--model-profile", "mock-paid"])  # no LLM: not paid


def test_disabled_profile_refused(paid_env):
    with pytest.raises(SystemExit):
        run_cli.parse_args(["--agent", "react", "--model-profile", "mock-off"])


def test_cli_paid_run_end_to_end_and_blocked(paid_env, capsys):
    assert run_cli.main(_cli(paid_env, "--allow-paid", "--max-usd", "0.5")) == 0
    out = capsys.readouterr().out
    assert "[paid] mock-paid" in out and "conservative default" in out and "budget OK" in out
    assert [json.loads(line)["type"] for line in paid_env.ledger.read_text().splitlines()] == \
        ["reservation", "final"]
    L = _ledger(paid_env)
    L.reserve(paid_env.out / "big", "mock-paid", "m", 18.9)   # leaves < $0.50 usable
    n_before = len(paid_env.ledger.read_text().splitlines())
    assert run_cli.main(_cli(paid_env, "--allow-paid", "--max-usd", "0.5")) == 2
    assert "Refusing to start a paid run" in capsys.readouterr().err
    assert len(paid_env.ledger.read_text().splitlines()) == n_before   # nothing reserved


def test_shipped_profiles_guardrails():
    assert resolve_profile("haiku-4.5").disabled is True
    h = resolve_profile("haiku-5.5")
    assert h.is_paid and not h.disabled and h.max_output_tokens == 384
    assert h.options["max_prompt_tokens"] == 90000 and h.options["send_temperature"] is False
    assert h.options["extra_body"]["thinking"] == {"type": "disabled"}
    assert h.pricing_date == "2026-10-21" and "claude-haiku-5-5" in h.pricing_source
    assert resolve_profile(backend="anthropic").name == "haiku-5.5"
    assert spend.load_budget().limit == pytest.approx(19.0)


# ---------------------------------------------------------------- Anthropic error policy
def _status_error(cls, code):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(f"HTTP {code}", response=httpx2.Response(code, request=req), body=None)


def _client(script):
    """script items: an exception to raise, or 'ok'."""
    calls = []

    def create(**kw):
        calls.append(kw)
        item = script.pop(0)
        if item != "ok":
            raise item
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Thought: x\nAction: wait")],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=7))
    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def _haiku(client):
    p = resolve_profile("haiku-5.5")
    m = AnthropicLLM(p.model, p.temperature, p.max_output_tokens, p.context_window, p.options, client=client)
    m.RETRY_DELAYS_S = (0,)
    return m


def test_anthropic_retries_429_5xx_and_connection_then_succeeds():
    import anthropic
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client, calls = _client([_status_error(anthropic.RateLimitError, 429),
                             _status_error(anthropic.InternalServerError, 500),
                             anthropic.APIConnectionError(request=req), "ok"])
    m = _haiku(client)
    r = m.complete("SYS", "USER")
    assert (r.input_tokens, r.output_tokens) == (100, 7) and len(calls) == 4
    assert m.consecutive_errors == 0
    assert calls[0]["max_tokens"] == 384 and "temperature" not in calls[0].get("extra_body", {})


def test_anthropic_never_retries_validation_errors():
    import anthropic
    client, calls = _client([_status_error(anthropic.BadRequestError, 400), "ok"])
    with pytest.raises(PaidAPIError, match="not retrying.*HTTP 400"):
        _haiku(client).complete("s", "u")
    assert len(calls) == 1


def test_anthropic_stops_after_5_consecutive_errors():
    import anthropic
    client, calls = _client([_status_error(anthropic.InternalServerError, 503)] * 9)
    with pytest.raises(PaidAPIError, match="5 consecutive"):
        _haiku(client).complete("s", "u")
    assert len(calls) == 5


def test_anthropic_prompt_size_abort_and_missing_usage():
    client, calls = _client(["ok"])
    with pytest.raises(PaidAPIError, match="max_prompt_tokens=90000"):
        _haiku(client).complete("x" * 400_000, "u")
    assert calls == []

    def create(**kw):
        return SimpleNamespace(content=[], usage=None)
    with pytest.raises(PaidAPIError, match="no usage"):
        _haiku(SimpleNamespace(messages=SimpleNamespace(create=create))).complete("s", "u")
