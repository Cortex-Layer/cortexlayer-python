"""Pluggable chat-model backends, shared by fact extraction and the reader.

Used by both ``facts/backends.py`` (ingest-time extraction, ``FactEngine._llm``)
and ``compression.py`` (query-time reader) — one provider abstraction, one place
to add a new model. ``facts/backends.py`` re-exports the names below for
backward compatibility; new code should import from here directly.

Task 0099: added ``CloudLLM`` plus named presets so a deployment can point
extraction and/or the reader at a cloud API (Muse Spark today, DeepSeek or
others later) instead of local Ollama, without either engine module knowing
which provider it's talking to.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Protocol

import httpx

from ..errors import CortexConfigError, LLMError

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3.5:9b"


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str:
        """Return the model's raw reply (expected to be a JSON object string)."""
        ...


class OllamaLLM:
    """Chat model over Ollama's ``/api/chat`` with native JSON output."""

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        host: Optional[str] = None,
        *,
        temperature: float = 0.1,
        top_p: float = 0.1,
        max_tokens: int = 2000,
        think: bool = False,
        timeout: float = 300.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.model = model
        self.host = (host or ollama_host()).rstrip("/")
        self._opts = {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
        self._think = think
        self._timeout = timeout
        self._http = http_client

    def generate(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                # Same nudge Mem0's Ollama client appends for JSON mode.
                {"role": "user", "content": user + "\n\nPlease respond with valid JSON only."},
            ],
            "format": "json",
            "stream": False,
            # Thinking models otherwise return a trace and JSON extraction
            # silently yields nothing (verified with qwen3.5:9b).
            "think": self._think,
            "options": self._opts,
        }
        try:
            client = self._http or httpx.Client(timeout=self._timeout)
            try:
                resp = client.post(f"{self.host}/api/chat", json=body)
            finally:
                if self._http is None:
                    client.close()
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMError(f"LLM call to {self.host} ({self.model}) failed: {e}") from e


#: Known OpenAI-compatible cloud chat providers: base URL, default model, and
#: any extra per-request fields the provider needs. Add a new provider here —
#: ``CloudLLM`` and ``resolve_llm`` need no changes.
#:
#: Muse Spark: contributor tier (cheaper, Meta trains on prompts/completions
#: sent to it — a real tradeoff for real user data, not just benchmark data;
#: flagged explicitly in task 0099) always reasons, so ``reasoning_effort``
#: must be set ("none" is rejected) — "minimal" is the cheapest valid value
#: and showed no correctness change on spot checks (task 0096).
#: DeepSeek: not wired as a default provider yet (task 0099 shipped Muse
#: Spark first), but the preset is here so switching is a config change, not
#: new code — matches ``run_locomo.py``'s ``deepseek_chat``.
#: OpenAI (gpt-5-nano): verified live 2026-09-27. Two real quirks baked into
#: this preset (both confirmed against the actual API, not assumed):
#: (1) rejects ``temperature=0`` — "Only the default (1) value is supported"
#: (400) — so ``temperature`` defaults to 1 here, unlike every other preset.
#: (2) always reasons like Muse Spark, and worse by default: an unset
#: ``reasoning_effort`` burned 256 reasoning tokens / 2.67s on a trivial
#: one-word JSON reply; ``"minimal"`` cut that to 0 reasoning tokens / 0.62s
#: with the same correct output — so it's the default here too, not opt-in.
CLOUD_PROVIDERS: dict[str, dict[str, Any]] = {
    "muse": {
        "base_url": "https://api.meta.ai/v1/chat/completions",
        "model": "muse-spark-1.3-contributor",
        "api_key_env": "MUSE_API_KEY",
        "extra_body": {"reasoning_effort": "minimal"},
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "extra_body": {},
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-5-nano",
        "api_key_env": "OPENAI_API_KEY",
        "extra_body": {"reasoning_effort": "minimal"},
        "temperature": 1,
    },
}


class CloudLLM:
    """Chat model over any OpenAI-compatible ``/chat/completions`` endpoint.

    Not tied to one vendor — ``base_url``/``model``/``extra_body`` are plain
    config, so a new cloud provider is a new entry in ``CLOUD_PROVIDERS``
    (or a one-off ``CloudLLM(...)`` for something not preset), never a new
    class.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        temperature: float = 0,
        extra_body: Optional[dict] = None,
        timeout: float = 120.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        if not api_key:
            raise CortexConfigError(f"CloudLLM({model!r}) needs an api_key")
        self.base_url = base_url
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._extra_body = dict(extra_body or {})
        self._timeout = timeout
        self._http = http_client

    def generate(self, system: str, user: str) -> str:
        messages = [{"role": "user", "content": user}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
            **self._extra_body,
        }
        try:
            client = self._http or httpx.Client(timeout=self._timeout)
            try:
                resp = client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=body,
                )
            finally:
                if self._http is None:
                    client.close()
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMError(f"LLM call to {self.base_url} ({self.model}) failed: {e}") from e


class _CallableLLM:
    def __init__(self, fn: Callable[[str, str], str]) -> None:
        self._fn = fn

    def generate(self, system: str, user: str) -> str:
        return self._fn(system, user)


def resolve_llm(spec: Any) -> LLM:
    """``None`` -> default Ollama; dict -> a provider (``"ollama"`` or a
    ``CLOUD_PROVIDERS`` key); an object with ``generate`` -> itself; a
    callable ``fn(system, user) -> str`` -> wrapped.

    Cloud dict spec: ``{"provider": "muse", "api_key": "...", ...}``.
    ``api_key`` defaults to ``os.environ[<provider's api_key_env>]`` if
    omitted. ``model``/``base_url``/``extra_body`` override that provider's
    preset; any other key is passed through to ``CloudLLM``.
    """
    if spec is None:
        return OllamaLLM()
    if isinstance(spec, dict):
        cfg = dict(spec)
        provider = cfg.pop("provider", "ollama")
        if provider == "ollama":
            return OllamaLLM(**cfg)
        if provider in CLOUD_PROVIDERS:
            preset = CLOUD_PROVIDERS[provider]
            api_key = cfg.pop("api_key", None) or os.environ.get(preset["api_key_env"])
            if not api_key:
                raise CortexConfigError(
                    f"llm provider {provider!r} needs api_key (arg) or "
                    f"${preset['api_key_env']}"
                )
            kwargs = {
                "base_url": cfg.pop("base_url", preset["base_url"]),
                "model": cfg.pop("model", preset["model"]),
                "api_key": api_key,
                "extra_body": cfg.pop("extra_body", preset["extra_body"]),
            }
            # Preset-level CloudLLM kwarg defaults beyond the four always-present
            # keys above (e.g. openai's temperature=1 — it rejects 0). Any value
            # explicitly in cfg still wins.
            for key, value in preset.items():
                if key not in ("base_url", "model", "api_key_env", "extra_body"):
                    kwargs.setdefault(key, value)
            kwargs.update(cfg)
            return CloudLLM(**kwargs)
        if provider == "cloud":
            # Fully custom OpenAI-compatible endpoint, no preset.
            try:
                base_url = cfg.pop("base_url")
                model = cfg.pop("model")
                api_key = cfg.pop("api_key")
            except KeyError as e:
                raise CortexConfigError(
                    f"llm provider 'cloud' needs base_url, model and api_key (missing {e})"
                ) from e
            return CloudLLM(base_url=base_url, model=model, api_key=api_key, **cfg)
        raise CortexConfigError(
            f"unknown llm provider {provider!r} (known: 'ollama', 'cloud', "
            f"{', '.join(sorted(CLOUD_PROVIDERS))!s})"
        )
    if hasattr(spec, "generate"):
        return spec
    if callable(spec):
        return _CallableLLM(spec)
    raise CortexConfigError(f"llm must be None, a dict, a callable or an object with generate(); got {spec!r}")
