"""Thin LLM backends with uniform token accounting, plus the model registry.

All backends expose ``complete(system, user) -> LLMResponse`` plus ``backend``,
``model`` and ``context_limit`` (max prompt tokens, or None when the backend
does not need the client-side overflow guard).

Models are described by named profiles in ``configs/models.yaml`` (backend,
model id, endpoint, context window, prices, backend options). Adding a model is
one YAML entry; ``make_llm_from_profile`` builds the backend.
"""
from __future__ import annotations

import copy
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

MODELS_YAML = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
BACKENDS = ("ollama", "openai_compat", "anthropic", "mock")

DEFAULT_MODELS = {  # kept for the CLI help text; the profiles are authoritative
    "ollama": "qwen2.5:7b",
    "anthropic": "claude-haiku-4-5-20251001",
    "mock": "mock",
}


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLMSetupError(RuntimeError):
    """Raised with a plain-English fix when a backend cannot be used."""


def estimate_tokens(text: str) -> int:
    """Rough token estimate (characters / 3.5) used for the context guard."""
    return int(len(text) / 3.5) + 1


# ----------------------------------------------------------------------------
# Model registry
@dataclass
class ModelProfile:
    name: str
    backend: str
    model: str
    base_url: str | None = None
    context_window: int = 8192
    max_output_tokens: int = 512
    temperature: float = 0.7
    price_in_per_mtok: float | None = 0.0    # USD per million input tokens (None = unknown)
    price_out_per_mtok: float | None = 0.0
    options: dict = field(default_factory=dict)
    pricing_source: str | None = None
    pricing_date: str | None = None
    notes: str | None = None
    disabled: bool = False   # refused by the CLI; only re-enabled by editing models.yaml

    @property
    def is_paid(self) -> bool:
        return bool((self.price_in_per_mtok or 0) > 0 or (self.price_out_per_mtok or 0) > 0)

    def usd(self, input_tokens: int, output_tokens: int) -> float | None:
        if self.price_in_per_mtok is None or self.price_out_per_mtok is None:
            return None
        return (input_tokens * self.price_in_per_mtok + output_tokens * self.price_out_per_mtok) / 1e6

    def to_dict(self) -> dict:
        return asdict(self)


def load_profiles(path: Path | str | None = None) -> tuple[dict[str, ModelProfile], dict[str, str]]:
    """Parse models.yaml -> ({name: ModelProfile}, {backend: default profile name})."""
    path = MODELS_YAML if path is None else path
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = set(ModelProfile.__dataclass_fields__) - {"name"}
    profiles = {}
    for name, spec in (data.get("profiles") or {}).items():
        spec = dict(spec or {})
        unknown = set(spec) - known
        if unknown:
            raise ValueError(f"models.yaml profile {name!r}: unknown keys {sorted(unknown)}")
        for req in ("backend", "model"):
            if req not in spec:
                raise ValueError(f"models.yaml profile {name!r}: missing {req!r}")
        if spec["backend"] not in BACKENDS:
            raise ValueError(f"models.yaml profile {name!r}: backend must be one of {BACKENDS}")
        spec["options"] = dict(spec.get("options") or {})
        profiles[name] = ModelProfile(name=name, **spec)
    defaults = dict(data.get("defaults") or {})
    for backend, name in defaults.items():
        if name not in profiles:
            raise ValueError(f"models.yaml default for {backend!r} names unknown profile {name!r}")
    return profiles, defaults


def resolve_profile(profile_name: str | None = None, backend: str | None = None,
                    model: str | None = None, temperature: float | None = None,
                    context_window: int | None = None, max_output_tokens: int | None = None,
                    base_url: str | None = None, path: Path | str | None = None) -> ModelProfile:
    """Pick a profile (by name, else the backend's default) and apply CLI overrides.

    ``model`` overrides the model id. If the new id matches another profile of the
    same backend, that profile's prices and options are used; otherwise the price
    is unknown (None) and cost cannot be computed.
    """
    profiles, defaults = load_profiles(path)
    if profile_name:
        if profile_name not in profiles:
            raise LLMSetupError(f"Unknown --model-profile {profile_name!r}. Profiles in "
                                f"{path or MODELS_YAML}: {', '.join(sorted(profiles))}")
        prof = copy.deepcopy(profiles[profile_name])
    else:
        backend = backend or "ollama"
        if backend not in defaults:
            raise LLMSetupError(f"models.yaml has no default profile for backend {backend!r}")
        prof = copy.deepcopy(profiles[defaults[backend]])
    if model and model != prof.model:
        match = [p for p in profiles.values() if p.backend == prof.backend and p.model == model]
        if match:
            prof = copy.deepcopy(match[0])
        else:
            prof.name = f"{prof.name}+model={model}"
            prof.model = model
            prof.price_in_per_mtok = prof.price_out_per_mtok = None
            prof.pricing_source = None
            prof.notes = "model overridden on the command line; price unknown"
    if temperature is not None:
        prof.temperature = temperature
    if context_window is not None:
        prof.context_window = context_window
    if max_output_tokens is not None:
        prof.max_output_tokens = max_output_tokens
    if base_url is not None:
        prof.base_url = base_url
    return prof


# ----------------------------------------------------------------------------
# Shared retry/backoff for local servers that may restart mid-run
# total wait ~ 5+10+20+40+60+60 s (e.g. the Ollama app auto-updating)
RETRY_DELAYS_S = (5, 10, 20, 40, 60, 60)


def call_with_retries(fn, transient: tuple, give_up_msg: str, label: str, delays=RETRY_DELAYS_S):
    for attempt in range(len(delays) + 1):
        try:
            return fn()
        except transient as exc:
            if attempt == len(delays):
                raise LLMSetupError(give_up_msg) from exc
            delay = delays[attempt]
            print(f"[{label}] {type(exc).__name__}: {exc}; retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)


class PaidAPIError(LLMSetupError):
    """A paid API call failed in a way we refuse to work around. Stops the run."""


class AnthropicLLM:
    """Profile options: ``send_temperature`` (default true; Claude Haiku 5.5 rejects
    non-default sampling values), ``extra_body`` (merged into the request, e.g.
    ``{"thinking": {"type": "disabled"}}``) and ``max_prompt_tokens`` (hard abort above
    it; Haiku 5.5 bills a whole request 5x above 100k prompt tokens). anthropic 1.x has
    no ``temperature`` keyword, so temperature travels in ``extra_body`` too.

    Errors: SDK retries are off. 429 / 5xx / connection errors are retried with backoff;
    any other API error (4xx validation, auth, not found) raises ``PaidAPIError``
    immediately. ``MAX_CONSECUTIVE_ERRORS`` failed attempts in a row stop the run."""

    backend = "anthropic"
    RETRY_DELAYS_S = (2, 5, 10, 20)
    MAX_CONSECUTIVE_ERRORS = 5

    def __init__(self, model: str, temperature: float = 0.0, max_output_tokens: int = 512,
                 context_window: int | None = None, options: dict | None = None, client=None):
        self.options = dict(options or {})
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMSetupError(
                    "ANTHROPIC_API_KEY is not set. Set it in your shell before using an "
                    "anthropic profile, e.g. in PowerShell:\n\n    $env:ANTHROPIC_API_KEY = \"sk-ant-...\"\n")
            import anthropic
            client = anthropic.Anthropic(max_retries=0)  # our own retry policy below
        self.client = client
        self.consecutive_errors = 0
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.context_limit = int(0.9 * context_window) if context_window else None

    def request_kwargs(self, system: str, user: str) -> dict:
        extra = dict(self.options.get("extra_body") or {})
        if self.options.get("send_temperature", True):
            extra["temperature"] = self.temperature
        kw = dict(model=self.model, max_tokens=self.max_output_tokens, system=system,
                  messages=[{"role": "user", "content": user}])
        if extra:
            kw["extra_body"] = extra
        return kw

    @staticmethod
    def _retryable(exc) -> bool:
        import anthropic
        if isinstance(exc, anthropic.APIConnectionError):   # includes timeouts
            return True
        status = getattr(exc, "status_code", None)
        return isinstance(exc, anthropic.APIStatusError) and (status == 429 or (status or 0) >= 500)

    def complete(self, system: str, user: str) -> LLMResponse:
        import anthropic

        limit = self.options.get("max_prompt_tokens")
        est = estimate_tokens(system + user)
        if limit and est > limit:
            raise PaidAPIError(f"Estimated prompt size {est} tokens exceeds max_prompt_tokens={limit} "
                               "for this profile; refusing to send it (price tier / runaway prompt).")
        kwargs = self.request_kwargs(system, user)
        t0 = time.perf_counter()
        attempt = 0
        while True:
            try:
                resp = self.client.messages.create(**kwargs)
            except anthropic.APIError as exc:
                self.consecutive_errors += 1
                status = getattr(exc, "status_code", None)
                desc = f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "") + f": {exc}"
                if not self._retryable(exc):
                    raise PaidAPIError(f"Anthropic API rejected the request, not retrying: {desc}") from exc
                if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    raise PaidAPIError(f"{self.consecutive_errors} consecutive Anthropic API errors; "
                                       f"stopping the run. Last: {desc}") from exc
                delay = self.RETRY_DELAYS_S[min(attempt, len(self.RETRY_DELAYS_S) - 1)]
                attempt += 1
                print(f"[anthropic] {desc}; retrying in {delay}s", file=sys.stderr)
                time.sleep(delay)
                continue
            self.consecutive_errors = 0
            break
        usage = getattr(resp, "usage", None)
        if usage is None or getattr(usage, "input_tokens", None) is None:
            raise PaidAPIError("Anthropic response had no usage field; cannot account for cost.")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, int(usage.input_tokens), int(usage.output_tokens or 0),
                           time.perf_counter() - t0)


class OllamaLLM:
    backend = "ollama"
    RETRY_DELAYS_S = RETRY_DELAYS_S

    def __init__(self, model: str, temperature: float = 0.0, num_ctx: int = 8192,
                 seed: int = 0, host: str = "http://localhost:11434",
                 max_output_tokens: int = 512):
        import ollama

        self._ollama = ollama
        self.client = ollama.Client(host=host)
        self.model = model
        self.host = host
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        # The agent keeps prompts under 90% of num_ctx so Ollama never silently
        # truncates them (which would cut the system prompt first).
        self.context_limit = int(0.9 * num_ctx)
        self.check_ready()

    def check_ready(self):
        try:
            listing = self.client.list()
        except ConnectionError as exc:
            raise LLMSetupError(
                f"Cannot reach the Ollama server at {self.host}.\n"
                "Start it in another terminal with:\n\n    ollama serve\n") from exc
        names = {m.model for m in listing.models}
        wanted = self.model if ":" in self.model else self.model + ":latest"
        if wanted not in names:
            raise LLMSetupError(
                f"The Ollama model '{self.model}' is not downloaded.\n"
                f"Download it with:\n\n    ollama pull {self.model}\n")

    def complete(self, system: str, user: str) -> LLMResponse:
        import httpx

        def chat():
            try:
                return self.client.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    options={"temperature": self.temperature, "num_ctx": self.num_ctx,
                             "seed": self.seed, "num_predict": self.max_output_tokens},
                )
            except self._ollama.ResponseError as exc:
                if exc.status_code == 404:
                    raise LLMSetupError(
                        f"The Ollama model '{self.model}' is not available. Download it with:\n\n"
                        f"    ollama pull {self.model}\n") from exc
                raise

        t0 = time.perf_counter()
        resp = call_with_retries(
            chat, (ConnectionError, httpx.TransportError),
            f"Lost connection to the Ollama server at {self.host}. Restart it with:\n\n    ollama serve\n",
            "ollama", self.RETRY_DELAYS_S)
        return LLMResponse(resp.message.content or "", int(resp.prompt_eval_count or 0),
                           int(resp.eval_count or 0), time.perf_counter() - t0)


class _RetryableStatus(Exception):
    pass


class OpenAICompatLLM:
    """Any server speaking the OpenAI chat-completions API (vLLM, llama.cpp
    server, LM Studio, TGI). Profile options: ``api_key_env`` (env var holding a
    bearer token, optional), ``supports_seed`` (default true), ``timeout_s``
    (default 600), ``extra_body`` (merged into the request)."""

    backend = "openai_compat"
    RETRY_DELAYS_S = RETRY_DELAYS_S

    def __init__(self, model: str, base_url: str, temperature: float = 0.7,
                 context_window: int = 8192, max_output_tokens: int = 512, seed: int = 0,
                 options: dict | None = None, transport=None):
        import httpx

        self._httpx = httpx
        self.options = dict(options or {})
        self.model = model
        base = (base_url or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        if not base:
            raise LLMSetupError("openai_compat profiles need a base_url, e.g. http://localhost:8000")
        self.base_url = base
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.context_limit = int(0.9 * context_window)
        headers = {}
        key_env = self.options.get("api_key_env")
        if key_env:
            key = os.environ.get(key_env)
            if not key:
                raise LLMSetupError(f"This profile reads its API key from ${key_env}, which is not set.")
            headers["Authorization"] = f"Bearer {key}"
        self.client = httpx.Client(base_url=base, headers=headers, transport=transport,
                                   timeout=float(self.options.get("timeout_s", 600)))
        self.check_ready()

    def check_ready(self):
        httpx = self._httpx
        try:
            r = self.client.get("/v1/models", timeout=10)
        except httpx.TransportError as exc:
            raise LLMSetupError(
                f"Cannot reach the OpenAI-compatible server at {self.base_url}.\n"
                "Start it first, for example:\n\n"
                f"    vllm serve {self.model} --port 8000\n"
                "    llama-server -m <model.gguf> --port 8000\n"
                "or start the server in LM Studio, then check base_url in configs/models.yaml.\n") from exc
        if r.status_code in (401, 403):
            raise LLMSetupError(f"The server at {self.base_url} rejected the API key "
                                f"(HTTP {r.status_code}). Check options.api_key_env in the profile.")
        if r.status_code != 200:
            raise LLMSetupError(f"GET {self.base_url}/v1/models returned HTTP {r.status_code}; "
                                "is this an OpenAI-compatible server?")
        ids = [m.get("id") for m in (r.json().get("data") or [])]
        if self.model not in ids:
            raise LLMSetupError(
                f"The server at {self.base_url} does not serve model '{self.model}'.\n"
                f"It serves: {', '.join(map(str, ids)) or '(nothing)'}.\n"
                "Set `model:` in the profile to one of these ids.")

    def request_body(self, system: str, user: str) -> dict:
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": self.temperature, "max_tokens": self.max_output_tokens}
        if self.options.get("supports_seed", True):
            body["seed"] = self.seed
        body.update(self.options.get("extra_body") or {})
        return body

    def complete(self, system: str, user: str) -> LLMResponse:
        body = self.request_body(system, user)

        def post():
            r = self.client.post("/v1/chat/completions", json=body)
            if r.status_code in (429, 502, 503, 504):
                raise _RetryableStatus(f"HTTP {r.status_code}")
            if r.status_code != 200:
                raise LLMSetupError(f"POST {self.base_url}/v1/chat/completions returned HTTP "
                                    f"{r.status_code}: {r.text[:300]}")
            return r.json()

        t0 = time.perf_counter()
        data = call_with_retries(
            post, (self._httpx.TransportError, _RetryableStatus),
            f"Lost connection to the OpenAI-compatible server at {self.base_url}. Restart it and rerun.",
            "openai_compat", self.RETRY_DELAYS_S)
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return LLMResponse(text, int(usage.get("prompt_tokens") or 0),
                           int(usage.get("completion_tokens") or 0), time.perf_counter() - t0)


class MockLLM:
    """Offline stand-in. Returns scripted replies in order (cycling), or, with no
    script, a seeded random well-formed reply. Token counts are estimates."""

    backend = "mock"

    def __init__(self, script: list[str] | None = None, seed: int = 0,
                 context_limit: int | None = None, model: str = "mock"):
        self.script = list(script) if script else None
        self.rng = random.Random(seed)
        self.calls: list[tuple[str, str]] = []
        self.context_limit = context_limit
        self.model = model
        self._i = 0

    def reply(self) -> str:
        return f"Thought: mock exploring.\nAction: {self.rng.randrange(5)}"

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        if self.script:
            text = self.script[self._i % len(self.script)]
            self._i += 1
        else:
            text = self.reply()
        return LLMResponse(text, estimate_tokens(system + user), estimate_tokens(text), 0.0)


class MockIntentLLM(MockLLM):
    """Mock for intent mode: seeded random well-formed intents (offline)."""

    INTENTS = ("right", "down", "left", "up", "wait", "wait(3)")

    def reply(self) -> str:
        r = self.rng.random()
        if r < 0.4:
            intent = f"interact({self.rng.randrange(6)},{self.rng.randrange(6)})"
        elif r < 0.7:
            intent = f"goto({self.rng.randrange(6)},{self.rng.randrange(6)})"
        else:
            intent = self.rng.choice(self.INTENTS)
        return f"Thought: mock exploring.\nAction: {intent}"


def make_llm_from_profile(profile: ModelProfile, seed: int, intent_mock: bool = False):
    b = profile.backend
    if b == "ollama":
        return OllamaLLM(profile.model, profile.temperature, profile.context_window, seed,
                         profile.base_url or "http://localhost:11434", profile.max_output_tokens)
    if b == "openai_compat":
        return OpenAICompatLLM(profile.model, profile.base_url, profile.temperature,
                               profile.context_window, profile.max_output_tokens, seed,
                               profile.options)
    if b == "anthropic":
        return AnthropicLLM(profile.model, profile.temperature, profile.max_output_tokens,
                            profile.context_window, profile.options)
    if b == "mock":
        cls = MockIntentLLM if intent_mock else MockLLM
        return cls(seed=seed, model=profile.model)
    raise ValueError(f"unknown backend {b!r}")


def make_llm(backend: str, model: str | None, temperature: float, seed: int,
             num_ctx: int = 8192, ollama_host: str = "http://localhost:11434",
             max_output_tokens: int = 512):
    """Backward-compatible constructor (pre-registry signature)."""
    prof = resolve_profile(backend=backend, model=model, temperature=temperature,
                           context_window=num_ctx, max_output_tokens=max_output_tokens,
                           base_url=ollama_host if backend == "ollama" else None)
    return make_llm_from_profile(prof, seed)


def fail(msg: str):  # pragma: no cover - CLI helper
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(2)
