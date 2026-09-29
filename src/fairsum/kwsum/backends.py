"""Generation backends behind one interface (spec §6).

Each backend turns a list of prompts into a list of Completions in the same
order. Local models go through `fairsum.generation.BatchedGenerator`; API
models go through their SDKs. Keys are read from the environment only and are
never written to config files or logs.

API completions carry the provider's stop reason, because for API models the
token count is not a truncation signal (thinking tokens are billed as output)
and a refusal must be recorded as a refusal. No server-side fallback to
another model is ever enabled: a silent model switch would confound the model
comparison the study makes.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    #: The provider's stop reason (API models); None for local generation.
    stop_reason: str | None = None


@dataclass(frozen=True)
class DecodingConfig:
    max_new_tokens: int = 256
    repetition_penalty: float = 1.1
    batch_size: int = 8


class Backend(Protocol):
    def complete(self, prompts: Sequence[str]) -> list[Completion]: ...


def _require_key(env: str) -> str:
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(
            f"environment variable {env} is not set; export the API key before running this model"
        )
    return key


class HFBackend:
    """A local model with REFER's settings: greedy, repetition penalty 1.1."""

    def __init__(self, model_id: str, decoding: DecodingConfig, load_in_4bit: bool = False,
                 dtype: str = "auto"):
        from fairsum.generation import BatchedGenerator

        self.decoding = decoding
        self._gen = BatchedGenerator(model_id, dtype=dtype, load_in_4bit=load_in_4bit)

    def complete(self, prompts: Sequence[str]) -> list[Completion]:
        out = self._gen.generate(
            list(prompts),
            k=1,
            batch_size=self.decoding.batch_size,
            max_new_tokens=self.decoding.max_new_tokens,
            greedy=True,
            repetition_penalty=self.decoding.repetition_penalty,
        )
        return [Completion(text, n_prompt, n_completion)
                for ((text, n_prompt, n_completion),) in out]


class _ThreadedAPIBackend:
    max_workers: int = 8

    def _one(self, prompt: str) -> Completion:
        raise NotImplementedError

    def complete(self, prompts: Sequence[str]) -> list[Completion]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._one, prompts))


class OpenAICompatBackend(_ThreadedAPIBackend):
    """OpenAI, or any OpenAI-compatible endpoint via `base_url` (OpenRouter, Groq).

    The chat API has no repetition penalty, so none is sent; this deviation from
    REFER's settings is recorded in the spec (§6). `temperature=None` omits the
    parameter for models that reject it; `token_param` exists because newer
    OpenAI models take `max_completion_tokens` instead of `max_tokens`.
    """

    def __init__(self, model: str, decoding: DecodingConfig, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", client: Any = None, max_workers: int = 8,
                 token_param: str = "max_tokens", temperature: float | None = 0):
        self.model = model
        self.decoding = decoding
        self.max_workers = max_workers
        self.token_param = token_param
        self.temperature = temperature
        if client is None:
            key = _require_key(api_key_env)
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=base_url, max_retries=5)
        self._client = client

    def _one(self, prompt: str) -> Completion:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            self.token_param: self.decoding.max_new_tokens,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        resp = self._client.chat.completions.create(**params)
        choice = resp.choices[0]
        return Completion((choice.message.content or "").strip(), int(resp.usage.prompt_tokens),
                          int(resp.usage.completion_tokens), getattr(choice, "finish_reason", None))


class AnthropicBackend(_ThreadedAPIBackend):
    """Claude models through the Messages API. Needs `pip install anthropic`.

    Current Claude models reject sampling parameters (temperature, top_p) with
    a 400, so none are sent unless `temperature` is given for an older model
    that accepts it. Claude Opus 5 thinks by default and thinking tokens count
    against `max_tokens`, so the cap is 4096, not the local 256: summary length
    is set by the prompt, and truncation is read from `stop_reason`.
    """

    def __init__(self, model: str, decoding: DecodingConfig,
                 api_key_env: str = "ANTHROPIC_API_KEY", client: Any = None, max_workers: int = 4,
                 max_tokens: int = 4096, temperature: float | None = None,
                 effort: str | None = None):
        self.model = model
        self.decoding = decoding
        self.max_workers = max_workers
        self.max_tokens = max(max_tokens, decoding.max_new_tokens)
        self.temperature = temperature
        self.effort = effort
        if client is None:
            key = _require_key(api_key_env)
            import anthropic

            client = anthropic.Anthropic(api_key=key, max_retries=5)
        self._client = client

    def _one(self, prompt: str) -> Completion:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.effort is not None:
            params["output_config"] = {"effort": self.effort}
        msg = self._client.messages.create(**params)
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return Completion(text.strip(), int(msg.usage.input_tokens), int(msg.usage.output_tokens),
                          msg.stop_reason)


def build_backend(spec: Any, decoding: DecodingConfig) -> Backend:
    if spec.backend == "hf":
        return HFBackend(spec.model_id, decoding, load_in_4bit=spec.load_in_4bit)
    if spec.backend == "openai":
        return OpenAICompatBackend(spec.model_id, decoding, base_url=spec.base_url,
                                   api_key_env=spec.api_key_env or "OPENAI_API_KEY",
                                   max_workers=spec.max_workers)
    if spec.backend == "anthropic":
        return AnthropicBackend(spec.model_id, decoding,
                                api_key_env=spec.api_key_env or "ANTHROPIC_API_KEY",
                                max_workers=spec.max_workers)
    raise ValueError(f"unknown backend {spec.backend!r}; expected hf, openai or anthropic")
