"""Thin LLM backends with uniform token accounting.

All backends expose ``complete(system, user) -> LLMResponse`` plus ``backend``,
``model`` and ``context_limit`` (max prompt tokens, or None when the backend
does not need the client-side overflow guard).
"""
from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass

DEFAULT_MODELS = {
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


class AnthropicLLM:
    backend = "anthropic"
    context_limit = None

    def __init__(self, model: str, temperature: float = 0.0, max_output_tokens: int = 512):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMSetupError(
                "ANTHROPIC_API_KEY is not set. Export it in your shell before using --llm anthropic.")
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def complete(self, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, resp.usage.input_tokens, resp.usage.output_tokens,
                           time.perf_counter() - t0)


class OllamaLLM:
    backend = "ollama"

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

    # Transient server restarts (e.g. the Ollama app auto-updating mid-run) are
    # retried with backoff; total wait ~ 5+10+20+40+60+60 s.
    RETRY_DELAYS_S = (5, 10, 20, 40, 60, 60)

    def complete(self, system: str, user: str) -> LLMResponse:
        import httpx

        t0 = time.perf_counter()
        for attempt in range(len(self.RETRY_DELAYS_S) + 1):
            try:
                resp = self.client.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    options={"temperature": self.temperature, "num_ctx": self.num_ctx,
                             "seed": self.seed, "num_predict": self.max_output_tokens},
                )
                break
            except (ConnectionError, httpx.TransportError) as exc:
                if attempt == len(self.RETRY_DELAYS_S):
                    raise LLMSetupError(
                        f"Lost connection to the Ollama server at {self.host}. "
                        "Restart it with:\n\n    ollama serve\n") from exc
                delay = self.RETRY_DELAYS_S[attempt]
                print(f"[ollama] {type(exc).__name__}: {exc}; retrying in {delay}s", file=sys.stderr)
                time.sleep(delay)
            except self._ollama.ResponseError as exc:
                if exc.status_code == 404:
                    raise LLMSetupError(
                        f"The Ollama model '{self.model}' is not available. Download it with:\n\n"
                        f"    ollama pull {self.model}\n") from exc
                raise
        return LLMResponse(resp.message.content or "", int(resp.prompt_eval_count or 0),
                           int(resp.eval_count or 0), time.perf_counter() - t0)


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

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        if self.script:
            text = self.script[self._i % len(self.script)]
            self._i += 1
        else:
            text = f"Thought: mock exploring.\nAction: {self.rng.randrange(5)}"
        return LLMResponse(text, estimate_tokens(system + user), estimate_tokens(text), 0.0)


def make_llm(backend: str, model: str | None, temperature: float, seed: int,
             num_ctx: int = 8192, ollama_host: str = "http://localhost:11434",
             max_output_tokens: int = 512):
    model = model or DEFAULT_MODELS[backend]
    if backend == "ollama":
        return OllamaLLM(model, temperature, num_ctx, seed, ollama_host, max_output_tokens)
    if backend == "anthropic":
        return AnthropicLLM(model, temperature, max_output_tokens)
    if backend == "mock":
        return MockLLM(seed=seed, model=model)
    raise ValueError(f"unknown backend {backend!r}")


def fail(msg: str):  # pragma: no cover - CLI helper
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(2)
