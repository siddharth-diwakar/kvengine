"""Hand-written greedy decode loops, one per phase of cache ownership.

Three implementations of the same thing, each owning more of the machinery:

  greedy_no_cache    - re-runs the whole sequence through the model every step.
                       Obviously correct, obviously quadratic. The reference the
                       cached paths are checked against.
  greedy_with_cache  - runs the prompt once, then feeds one token per step and
                       passes HuggingFace's past_key_values object forward.
  greedy_own_cache   - same loop, but the history lives in a tensor we allocated
                       and is read by attention we wrote (phase 1).
  greedy_paged       - same loop again, with the history scattered across
                       fixed-size blocks handed out by an allocator (phase 2).

All four must emit identical tokens. When one diverges, the difference between it
and the next one down tells you which layer of ownership broke.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class GenerationResult:
    prompt_token_ids: list[int]
    new_token_ids: list[int]
    elapsed_s: float
    stopped_on_eos: bool
    # Last-position logits for each decode step, [steps, vocab] on CPU in fp32.
    # Only populated when collect_logits=True; used to compare decode paths
    # numerically instead of only by their final token ids.
    step_logits: torch.Tensor | None = field(default=None, repr=False)
    # Fraction of reserved cache positions that ended up holding real data.
    # Only set by paths that manage their own cache.
    cache_utilization: float | None = None
    # Paged path only: blocks the request held at completion, and its sequence
    # handle when the caller asked to keep the blocks for inspection.
    blocks_held: int | None = None
    sequence: object | None = field(default=None, repr=False)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.new_token_ids

    @property
    def tokens_per_s(self) -> float:
        return len(self.new_token_ids) / self.elapsed_s if self.elapsed_s else 0.0


# Sentinel for "take the stop token from the model's own config", which is what
# .generate() does. Passing eos_token_id=None instead means "never stop early",
# useful when a test needs an exact, fixed number of decode steps.
AUTO_EOS = "auto"


def _resolve_eos(model, eos_token_id) -> set[int]:
    if eos_token_id == AUTO_EOS:
        eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    return _normalize_eos(eos_token_id)


def _normalize_eos(eos_token_id) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return set(eos_token_id)


def _sync(device: torch.device) -> None:
    """Make timings honest: MPS and CUDA queue work asynchronously."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def greedy_no_cache(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    eos_token_id=AUTO_EOS,
    collect_logits: bool = False,
) -> GenerationResult:
    """Greedy decode with no cache at all: full forward pass every step."""
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "batch size 1 only in phase 0"
    device = input_ids.device
    eos_ids = _resolve_eos(model, eos_token_id)

    prompt = input_ids[0].tolist()
    seq = input_ids
    new_tokens: list[int] = []
    logits_log: list[torch.Tensor] = []
    stopped = False

    _sync(device)
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        out = model(input_ids=seq, use_cache=False)
        next_logits = out.logits[0, -1, :]
        if collect_logits:
            logits_log.append(next_logits.detach().float().cpu())
        next_id = int(torch.argmax(next_logits))
        new_tokens.append(next_id)
        seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id in eos_ids:
            stopped = True
            break
    _sync(device)

    return GenerationResult(
        prompt_token_ids=prompt,
        new_token_ids=new_tokens,
        elapsed_s=time.perf_counter() - t0,
        stopped_on_eos=stopped,
        step_logits=torch.stack(logits_log) if logits_log else None,
    )


@torch.no_grad()
def greedy_with_cache(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    eos_token_id=AUTO_EOS,
    collect_logits: bool = False,
) -> GenerationResult:
    """Greedy decode reusing HuggingFace's KV cache between steps.

    Structure of the loop mirrors what every real serving engine does:
      1. prefill  - one forward pass over the whole prompt, cache is populated
      2. decode   - one forward pass per token, feeding only the newest token

    The model needs an attention_mask spanning prompt + everything generated so
    far, even though only one token is passed in, because the cache holds the
    rest of the sequence.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "batch size 1 only in phase 0"
    device = input_ids.device
    eos_ids = _resolve_eos(model, eos_token_id)

    prompt = input_ids[0].tolist()
    new_tokens: list[int] = []
    logits_log: list[torch.Tensor] = []
    stopped = False

    _sync(device)
    t0 = time.perf_counter()

    # --- prefill ---
    attn_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
    cache = out.past_key_values
    next_logits = out.logits[0, -1, :]

    for _ in range(max_new_tokens):
        if collect_logits:
            logits_log.append(next_logits.detach().float().cpu())
        next_id = int(torch.argmax(next_logits))
        new_tokens.append(next_id)
        if next_id in eos_ids:
            stopped = True
            break

        # --- decode: one token in, cache carries the history ---
        attn_mask = torch.cat(
            [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=device)], dim=1
        )
        out = model(
            input_ids=torch.tensor([[next_id]], device=device),
            attention_mask=attn_mask,
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        next_logits = out.logits[0, -1, :]
    _sync(device)

    return GenerationResult(
        prompt_token_ids=prompt,
        new_token_ids=new_tokens,
        elapsed_s=time.perf_counter() - t0,
        stopped_on_eos=stopped,
        step_logits=torch.stack(logits_log) if logits_log else None,
    )


@torch.no_grad()
def greedy_own_cache(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    eos_token_id=AUTO_EOS,
    collect_logits: bool = False,
    cache=None,
    max_seq_len: int | None = None,
) -> GenerationResult:
    """Phase 1: greedy decode against a KV cache we allocate and index ourselves.

    Identical loop shape to greedy_with_cache — prefill once, then one token per
    step. The difference is that no HuggingFace Cache object is involved: our
    own tensor holds the history, and our own attention reads it.

    Passing `cache` in lets a caller reuse one buffer across requests, which is
    what a real server does and what makes the reservation waste visible.
    """
    from .cache import ContiguousKVCache
    from .forward import forward_with_own_cache

    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "batch size 1 only in phase 1"
    device = input_ids.device
    eos_ids = _resolve_eos(model, eos_token_id)

    prompt = input_ids[0].tolist()
    if cache is None:
        room = max_seq_len if max_seq_len is not None else len(prompt) + max_new_tokens
        cache = ContiguousKVCache.for_model(model, max_seq_len=room)

    new_tokens: list[int] = []
    logits_log: list[torch.Tensor] = []
    stopped = False

    _sync(device)
    t0 = time.perf_counter()

    # --- prefill ---
    logits = forward_with_own_cache(model, input_ids, cache)
    next_logits = logits[0, -1, :]

    for _ in range(max_new_tokens):
        if collect_logits:
            logits_log.append(next_logits.detach().float().cpu())
        next_id = int(torch.argmax(next_logits))
        new_tokens.append(next_id)
        if next_id in eos_ids:
            stopped = True
            break

        # --- decode: one token, cache supplies the history ---
        logits = forward_with_own_cache(
            model, torch.tensor([[next_id]], device=device), cache
        )
        next_logits = logits[0, -1, :]
    _sync(device)

    return GenerationResult(
        prompt_token_ids=prompt,
        new_token_ids=new_tokens,
        elapsed_s=time.perf_counter() - t0,
        stopped_on_eos=stopped,
        step_logits=torch.stack(logits_log) if logits_log else None,
        cache_utilization=cache.utilization(),
    )


@torch.no_grad()
def greedy_paged(
    model,
    input_ids: torch.Tensor,
    pool,
    max_new_tokens: int = 32,
    eos_token_id=AUTO_EOS,
    collect_logits: bool = False,
    request_id: object = "req",
    free_on_finish: bool = True,
) -> GenerationResult:
    """Phase 2: greedy decode against the paged pool.

    Byte-identical loop to greedy_own_cache. The only change is the cache handed
    to the forward pass, which is the point: paging is invisible above write().

    Blocks are returned to the pool on completion unless free_on_finish=False,
    which the tests use to inspect the block table after the fact.
    """
    from .forward import forward_with_own_cache

    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "one request at a time"
    device = input_ids.device
    eos_ids = _resolve_eos(model, eos_token_id)

    prompt = input_ids[0].tolist()
    seq = pool.new_sequence(request_id)

    new_tokens: list[int] = []
    logits_log: list[torch.Tensor] = []
    stopped = False

    _sync(device)
    t0 = time.perf_counter()
    try:
        # --- prefill ---
        logits = forward_with_own_cache(model, input_ids, seq)
        next_logits = logits[0, -1, :]

        for _ in range(max_new_tokens):
            if collect_logits:
                logits_log.append(next_logits.detach().float().cpu())
            next_id = int(torch.argmax(next_logits))
            new_tokens.append(next_id)
            if next_id in eos_ids:
                stopped = True
                break

            # --- decode: blocks grow one at a time, as needed ---
            logits = forward_with_own_cache(
                model, torch.tensor([[next_id]], device=device), seq
            )
            next_logits = logits[0, -1, :]
        _sync(device)

        elapsed = time.perf_counter() - t0
        utilization = seq.utilization()
        blocks_held = len(seq.blocks)
    finally:
        if free_on_finish:
            seq.free()

    return GenerationResult(
        prompt_token_ids=prompt,
        new_token_ids=new_tokens,
        elapsed_s=elapsed,
        stopped_on_eos=stopped,
        step_logits=torch.stack(logits_log) if logits_log else None,
        cache_utilization=utilization,
        blocks_held=blocks_held,
        sequence=None if free_on_finish else seq,
    )
