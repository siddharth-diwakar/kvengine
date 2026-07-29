"""The HuggingFace reference we measure correctness against.

Subtlety worth knowing: `model.generate(do_sample=False)` is NOT necessarily
pure argmax. Qwen ships a generation_config.json carrying sampling defaults
(temperature, top_p, top_k, repetition_penalty). Warpers like
repetition_penalty are applied regardless of do_sample, so a naive
`.generate(do_sample=False)` can diverge from a plain argmax loop and it looks
exactly like a cache bug.

So the reference builds an explicit GenerationConfig with every logits
processor disabled. That makes "my loop matches .generate()" a statement about
the KV cache, not about undocumented defaults.
"""

from __future__ import annotations

import time

import torch
from transformers import GenerationConfig

from .decode import GenerationResult, _normalize_eos, _sync


def strict_greedy_config(model, max_new_tokens: int, eos_token_id=None) -> GenerationConfig:
    """A GenerationConfig that is pure argmax and nothing else."""
    if eos_token_id is None:
        eos_token_id = model.generation_config.eos_token_id
    return GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        min_new_tokens=0,
        temperature=None,
        top_p=None,
        top_k=None,
        typical_p=None,
        repetition_penalty=1.0,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        renormalize_logits=False,
        eos_token_id=eos_token_id,
        pad_token_id=model.generation_config.pad_token_id or eos_token_id,
        use_cache=True,
    )


@torch.no_grad()
def hf_greedy(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    eos_token_id=None,
) -> GenerationResult:
    """Run HuggingFace `.generate()` under strict greedy settings."""
    device = input_ids.device
    cfg = strict_greedy_config(model, max_new_tokens, eos_token_id)
    eos_ids = _normalize_eos(cfg.eos_token_id)

    _sync(device)
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        generation_config=cfg,
    )
    _sync(device)
    elapsed = time.perf_counter() - t0

    prompt_len = input_ids.shape[1]
    new_tokens = out[0, prompt_len:].tolist()
    return GenerationResult(
        prompt_token_ids=input_ids[0].tolist(),
        new_token_ids=new_tokens,
        elapsed_s=elapsed,
        stopped_on_eos=bool(new_tokens) and new_tokens[-1] in eos_ids,
    )
