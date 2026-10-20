"""Model registry, openai_compat backend (mocked transport), Anthropic request shape,
and the --max-usd budget. All offline."""
import inspect
import json
from types import SimpleNamespace

import httpx
import pytest

from react_baseline import run as run_cli
from react_baseline.agents import ReActAgent
from react_baseline.llm import (AnthropicLLM, LLMSetupError, MockLLM, OpenAICompatLLM,
                                load_profiles, make_llm_from_profile, resolve_profile)
from react_baseline.runner import Budgets, run


def test_shipped_yaml_parses():
    profiles, defaults = load_profiles()
    assert defaults["ollama"] == "qwen2.5-7b" and defaults["mock"] == "mock"
    assert {"qwen2.5-7b", "qwen2.5-14b", "vllm-qwen2.5-7b", "haiku-4.5", "haiku-5.5"} <= set(profiles)
    for p in profiles.values():
        assert p.context_window > 0 and p.max_output_tokens > 0
        if p.backend in ("ollama", "openai_compat", "mock"):
            assert p.price_in_per_mtok == 0 and p.price_out_per_mtok == 0
        if p.backend == "anthropic":
            assert p.price_in_per_mtok > 0 and p.pricing_source and p.pricing_date
    assert profiles["vllm-qwen2.5-7b"].backend == "openai_compat"


def test_new_model_is_one_yaml_entry(tmp_path):
    y = tmp_path / "models.yaml"
    y.write_text("defaults: {mock: m}\nprofiles:\n  m: {backend: mock, model: mock}\n"
                 "  my-llamacpp:\n    backend: openai_compat\n    model: my-gguf\n"
                 "    base_url: http://127.0.0.1:8080/v1\n    context_window: 4096\n"
                 "    price_in_per_mtok: 0.5\n    price_out_per_mtok: 1.5\n"
                 "    options: {supports_seed: false}\n", encoding="utf-8")
    p = resolve_profile("my-llamacpp", path=y)
    assert (p.backend, p.model, p.context_window) == ("openai_compat", "my-gguf", 4096)
    assert p.options == {"supports_seed": False}
    assert p.usd(1_000_000, 1_000_000) == pytest.approx(2.0)
    y.write_text("profiles:\n  bad: {backend: mock, model: x, colour: red}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_profiles(y)


def test_resolve_overrides_and_unknown_price():
    p = resolve_profile(backend="ollama")
    assert p.name == "qwen2.5-7b" and p.temperature == 0.7 and p.context_window == 8192
    p = resolve_profile(backend="ollama", model="qwen2.5:14b")
    assert p.name == "qwen2.5-14b"
    p = resolve_profile(backend="ollama", model="some-new-model:3b", temperature=0.2, context_window=4096)
    assert p.model == "some-new-model:3b" and p.usd(10, 10) is None
    assert p.temperature == 0.2 and p.context_window == 4096
    with pytest.raises(LLMSetupError, match="Unknown --model-profile"):
        resolve_profile("nope")


def test_cli_profile_resolution():
    a = run_cli.parse_args(["--agent", "react", "--model-profile", "haiku-5.5"])
    assert a.llm == "anthropic" and a.profile.model == "claude-haiku-5-5"
    a = run_cli.parse_args(["--agent", "react", "--llm", "anthropic"])
    assert a.profile.name == "haiku-4.5"
    with pytest.raises(SystemExit):  # no price for an ad-hoc model id
        run_cli.parse_args(["--agent", "react", "--model", "mystery:1b", "--max-usd", "1"])
    with pytest.raises(SystemExit):  # backend conflict
        run_cli.parse_args(["--agent", "react", "--llm", "ollama", "--model-profile", "haiku-4.5"])


# ---------------------------------------------------------------- openai_compat
def _server(models=("served-model",), fail_first=0, seen=None):
    state = {"n": 0}

    def handler(request: httpx.Request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in models]})
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            if seen is not None:
                seen.append(body)
            state["n"] += 1
            if state["n"] <= fail_first:
                return httpx.Response(503, text="loading")
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Thought: hi\nAction: 1"}}],
                "usage": {"prompt_tokens": 321, "completion_tokens": 12}})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_openai_compat_complete_usage_and_seed():
    seen = []
    llm = OpenAICompatLLM("served-model", "http://fake:8000/v1", temperature=0.3, context_window=4000,
                          max_output_tokens=64, seed=7, transport=_server(seen=seen))
    assert llm.context_limit == 3600 and llm.base_url == "http://fake:8000"
    r = llm.complete("SYS", "USER")
    assert r.text.endswith("Action: 1") and (r.input_tokens, r.output_tokens) == (321, 12)
    body = seen[0]
    assert body["seed"] == 7 and body["temperature"] == 0.3 and body["max_tokens"] == 64
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    llm = OpenAICompatLLM("served-model", "http://fake:8000", seed=7,
                          options={"supports_seed": False}, transport=_server(seen=seen))
    llm.complete("s", "u")
    assert "seed" not in seen[-1]


def test_openai_compat_retries_then_succeeds():
    llm = OpenAICompatLLM("served-model", "http://fake:8000", transport=_server(fail_first=2))
    llm.RETRY_DELAYS_S = (0, 0, 0)
    assert llm.complete("s", "u").output_tokens == 12
    llm = OpenAICompatLLM("served-model", "http://fake:8000", transport=_server(fail_first=9))
    llm.RETRY_DELAYS_S = (0,)
    with pytest.raises(LLMSetupError, match="Lost connection"):
        llm.complete("s", "u")


def test_openai_compat_setup_errors():
    with pytest.raises(LLMSetupError, match="does not serve model 'wanted'"):
        OpenAICompatLLM("wanted", "http://fake:8000", transport=_server(models=("other",)))

    def down(request):
        raise httpx.ConnectError("refused", request=request)
    with pytest.raises(LLMSetupError, match="Cannot reach the OpenAI-compatible server"):
        OpenAICompatLLM("m", "http://fake:8000", transport=httpx.MockTransport(down))


# ---------------------------------------------------------------- anthropic shape
def _fake_client(captured):
    def create(**kw):
        captured.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Thought: x\nAction: 2")],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=10))
    return SimpleNamespace(messages=SimpleNamespace(create=create))


@pytest.mark.parametrize("name", ["haiku-4.5", "haiku-5.5"])
def test_anthropic_request_matches_installed_sdk(name):
    """anthropic 1.x has no `temperature` keyword; the request must bind to the real
    Messages.create signature, and Haiku 5.5 must not send a temperature."""
    from anthropic.resources.messages import Messages
    prof = resolve_profile(name)
    captured = []
    llm = AnthropicLLM(prof.model, prof.temperature, prof.max_output_tokens, prof.context_window,
                       prof.options, client=_fake_client(captured))
    r = llm.complete("SYS", "USER")
    assert (r.input_tokens, r.output_tokens) == (100, 10)
    kw = captured[0]
    inspect.signature(Messages.create).bind(None, **kw)  # raises TypeError on a bad keyword
    extra = kw.get("extra_body", {})
    if name == "haiku-4.5":
        assert extra.get("temperature") == 0.7
    else:
        assert "temperature" not in extra and extra["thinking"] == {"type": "disabled"}
    assert llm.context_limit == int(0.9 * prof.context_window)


def test_mock_profile_builds_offline():
    llm = make_llm_from_profile(resolve_profile("mock"), seed=0)
    assert isinstance(llm, MockLLM) and llm.backend == "mock"


# ---------------------------------------------------------------- --max-usd
def test_max_usd_stops_run(tmp_path):
    prof = resolve_profile("mock").to_dict()
    prof.update(price_in_per_mtok=1000.0, price_out_per_mtok=1000.0)  # ~$0.8 per mock call
    cfg = {"backend": "mock", "model": "mock", "model_profile": prof}
    s = run(ReActAgent(MockLLM()), "V2_task_01_tomato", 0,
            Budgets(max_env_steps=500, max_usd=3.0), tmp_path, cfg)
    assert s["stop_reason"] == "max_usd"
    per_call = s["usd_cost"] / s["llm_calls"]
    assert 3.0 <= s["usd_cost"] < 3.0 + 2 * per_call
    assert s["llm_calls"] < 500
    assert s["llm_calls_per_decision"] == 1.0
    assert s["llm_latency_median_s"] is not None


def test_usd_zero_for_local_profiles(tmp_path):
    cfg = {"backend": "mock", "model": "mock", "model_profile": resolve_profile("qwen2.5-7b").to_dict()}
    s = run(ReActAgent(MockLLM()), "V2_task_01_tomato", 0, Budgets(max_env_steps=5), tmp_path, cfg)
    assert s["usd_cost"] == 0.0 and s["total_tokens"] > 0
