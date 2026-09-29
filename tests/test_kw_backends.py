"""Backends: one interface for local and API models. No network, no GPU."""

import inspect
from types import SimpleNamespace

import pytest

from fairsum.generation import BatchedGenerator
from fairsum.kwsum.backends import (
    AnthropicBackend,
    Completion,
    DecodingConfig,
    OpenAICompatBackend,
    build_backend,
)

DEC = DecodingConfig(max_new_tokens=256, repetition_penalty=1.1, batch_size=8)


class FakeOpenAI:
    def __init__(self, finish_reason="stop"):
        self.calls = []
        self.finish_reason = finish_reason
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=f"  summary of {prompt} "),
                finish_reason=self.finish_reason,
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class FakeAnthropic:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=f"sum {prompt}"),
            ],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=11, output_tokens=6),
        )


class TestOpenAICompat:
    def test_preserves_order_and_strips_text(self):
        out = OpenAICompatBackend("gpt-x", DEC, client=FakeOpenAI(), max_workers=4).complete(["p1", "p2", "p3"])
        assert [o.text for o in out] == ["summary of p1", "summary of p2", "summary of p3"]
        assert out[0] == Completion("summary of p1", 10, 5, "stop")

    def test_sends_greedy_settings(self):
        client = FakeOpenAI()
        OpenAICompatBackend("gpt-x", DEC, client=client).complete(["p"])
        assert client.calls[0]["temperature"] == 0
        assert client.calls[0]["max_tokens"] == 256

    def test_temperature_can_be_omitted_for_models_that_reject_it(self):
        client = FakeOpenAI()
        OpenAICompatBackend("gpt-x", DEC, client=client, temperature=None).complete(["p"])
        assert "temperature" not in client.calls[0]

    def test_missing_key_names_the_variable(self, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NOPE_KEY"):
            OpenAICompatBackend("gpt-x", DEC, api_key_env="NOPE_KEY")


class TestAnthropic:
    def test_keeps_only_text_blocks_and_reads_usage_and_stop_reason(self):
        out = AnthropicBackend("claude-x", DEC, client=FakeAnthropic()).complete(["p1"])
        assert out == [Completion("sum p1", 11, 6, "end_turn")]

    def test_sends_no_sampling_parameters_by_default(self):
        client = FakeAnthropic()
        AnthropicBackend("claude-x", DEC, client=client).complete(["p"])
        assert "temperature" not in client.calls[0] and "top_p" not in client.calls[0]

    def test_token_cap_leaves_room_for_thinking(self):
        client = FakeAnthropic()
        AnthropicBackend("claude-x", DEC, client=client).complete(["p"])
        assert client.calls[0]["max_tokens"] == 4096

    def test_refusal_stop_reason_is_carried_through(self):
        out = AnthropicBackend("claude-x", DEC, client=FakeAnthropic("refusal")).complete(["p"])
        assert out[0].stop_reason == "refusal"

    def test_missing_key_names_the_variable(self, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NOPE_KEY"):
            AnthropicBackend("claude-x", DEC, api_key_env="NOPE_KEY")


def test_local_completions_have_no_stop_reason():
    assert Completion("t", 1, 2).stop_reason is None


def test_unknown_backend_is_rejected():
    spec = SimpleNamespace(backend="bogus", model_id="x", load_in_4bit=False,
                           base_url=None, api_key_env=None, max_workers=1)
    with pytest.raises(ValueError, match="bogus"):
        build_backend(spec, DEC)


def test_generate_accepts_a_repetition_penalty():
    params = inspect.signature(BatchedGenerator.generate).parameters
    assert "repetition_penalty" in params
    assert params["repetition_penalty"].default is None
