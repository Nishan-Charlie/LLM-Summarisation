"""Batched generation of k candidate summaries per instance.

v1's loop generated one prompt at a time -- 540 short generations in 50 minutes
on a 1B model, and 75 generations in 2h43m on an 8B with CPU offload. That is
why test-time scaling looked unaffordable; it is a batching problem, not a
compute problem.

The k samples of one instance share a prompt, so they are laid out contiguously
and batched together, letting the prefill be computed once per group.

Every generation is recorded with its token counts, including samples that are
later rejected: H2 compares parallel against sequential scaling per FLOP, and a
denominator that quietly omits discarded work would flatter both regimes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import pandas as pd

from fairsum.prompts import PROMPT_SPECS, parse_final_summary, requires_marker

RECORD_COLUMNS = [
    "instance_id",
    "method",
    "sample_idx",
    "raw",
    "summary",
    "parse_status",
    "prompt_tokens",
    "completion_tokens",
    "seed",
    "model",
]


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    prompt_idx: int
    sample_idx: int


@dataclass(frozen=True)
class GenerationRecord:
    instance_id: str
    method: str
    sample_idx: int
    raw: str
    summary: str | None
    parse_status: str
    prompt_tokens: int
    completion_tokens: int
    seed: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_batches(n_items: int, batch_size: int) -> list[tuple[int, int]]:
    """Half-open (start, stop) spans covering `n_items`."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [(s, min(s + batch_size, n_items)) for s in range(0, n_items, batch_size)]


def expand_requests(prompts: Sequence[str], k: int) -> list[GenerationRequest]:
    """Repeat each prompt k times, keeping a prompt's samples contiguous."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return [
        GenerationRequest(prompt=prompt, prompt_idx=p_idx, sample_idx=s_idx)
        for p_idx, prompt in enumerate(prompts)
        for s_idx in range(k)
    ]


def records_to_frame(records: Sequence[GenerationRecord]) -> pd.DataFrame:
    """Tabulate records, preserving the schema even when there are none."""
    if not records:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    return pd.DataFrame([r.to_dict() for r in records])[RECORD_COLUMNS]


class BatchedGenerator:
    """Wraps a causal LM for batched, seeded, k-sample generation.

    Imports torch lazily so the pure-logic modules stay importable on a machine
    with no GPU stack.
    """

    def __init__(
        self,
        model_id: str,
        dtype: str = "auto",
        load_in_4bit: bool = False,
        device_map: str = "auto",
        max_input_tokens: int = 8192,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def _as_chat(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        prompts: Sequence[str],
        k: int = 1,
        batch_size: int = 16,
        max_new_tokens: int = 320,
        temperature: float = 0.8,
        seed: int = 0,
        greedy: bool = False,
        repetition_penalty: float | None = None,
    ) -> list[list[tuple[str, int, int]]]:
        """Generate k samples per prompt.

        Returns, per prompt, a list of (raw_text, prompt_tokens, completion_tokens).
        Greedy decoding collapses k to 1 sample repeated, so it is only useful as
        the v1-equivalent baseline.
        """
        torch = self._torch
        requests = expand_requests(prompts, k)
        outputs: list[list[tuple[str, int, int]]] = [[] for _ in prompts]

        for start, stop in plan_batches(len(requests), batch_size):
            chunk = requests[start:stop]
            texts = [self._as_chat(r.prompt) for r in chunk]
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.model.device)

            torch.manual_seed(seed + start)
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=not greedy,
                    temperature=None if greedy else temperature,
                    top_p=None if greedy else 0.95,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            prompt_len = encoded["input_ids"].shape[1]
            completions = generated[:, prompt_len:]
            decoded = self.tokenizer.batch_decode(completions, skip_special_tokens=True)

            pad_id = self.tokenizer.pad_token_id
            for request, text, row, prompt_row in zip(
                chunk, decoded, completions, encoded["input_ids"]
            ):
                outputs[request.prompt_idx].append(
                    (
                        text.strip(),
                        int((prompt_row != pad_id).sum()),
                        int((row != pad_id).sum()),
                    )
                )
        return outputs

    def generate_records(
        self,
        instance_ids: Sequence[str],
        prompts: Sequence[str],
        method: str,
        seed: int = 0,
        **kwargs: Any,
    ) -> list[GenerationRecord]:
        """Generate and parse, keeping raw text and parse status for every sample."""
        raw_batches = self.generate(prompts, seed=seed, **kwargs)
        # Whether a missing marker is a failure depends on the method: prompts
        # with no reasoning scaffold have nothing to strip. See `PROMPT_SPECS`.
        needs_marker = requires_marker(method) if method in PROMPT_SPECS else True

        records: list[GenerationRecord] = []
        for instance_id, samples in zip(instance_ids, raw_batches):
            for sample_idx, (raw, n_prompt, n_completion) in enumerate(samples):
                parsed = parse_final_summary(raw, requires_marker=needs_marker)
                records.append(
                    GenerationRecord(
                        instance_id=instance_id,
                        method=method,
                        sample_idx=sample_idx,
                        raw=raw,
                        summary=parsed.summary,
                        parse_status=parsed.status.value,
                        prompt_tokens=n_prompt,
                        completion_tokens=n_completion,
                        seed=seed,
                        model=self.model_id,
                    )
                )
        return records
