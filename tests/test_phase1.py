"""Phase 1: our own KV cache must be indistinguishable from HuggingFace's.

Same correctness anchor as phase 0, one layer deeper: the history now lives in a
tensor we allocated, read by attention we wrote.
"""

import pytest
import torch

from kvengine import (
    ContiguousKVCache,
    KVCacheFull,
    forward_with_own_cache,
    greedy_own_cache,
    greedy_with_cache,
    hf_greedy,
)
from kvengine.forward import causal_mask, repeat_kv

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "Q: What is 17 * 3?\nA:",
]


# --- the anchor ---

def test_own_cache_matches_hf_generate(model, encode):
    for prompt in PROMPTS:
        ids = encode(prompt)
        mine = greedy_own_cache(model, ids, max_new_tokens=24)
        ref = hf_greedy(model, ids, max_new_tokens=24)
        assert mine.new_token_ids == ref.new_token_ids, f"diverged on prompt: {prompt!r}"


def test_own_cache_logits_match_hf_cache_logits(model, encode):
    """Compare scores, not just winning tokens.

    A cache that is subtly misaligned can still pick the same token when the top
    choice wins by a wide margin. Comparing the raw logits catches it earlier.
    """
    ids = encode(PROMPTS[0])
    mine = greedy_own_cache(model, ids, max_new_tokens=8, eos_token_id=None, collect_logits=True)
    theirs = greedy_with_cache(model, ids, max_new_tokens=8, eos_token_id=None, collect_logits=True)
    max_diff = (mine.step_logits - theirs.step_logits).abs().max().item()
    assert max_diff < 1e-3, f"our attention drifts from HF's: {max_diff}"


def test_prefill_logits_match_full_forward(model, encode):
    """Prefill in one shot must equal a plain no-cache forward over the prompt.

    This isolates the causal mask: if query/key positions are misaligned during
    prefill, every prompt position but the last is quietly wrong, and greedy
    decoding would not necessarily notice.
    """
    ids = encode("The quick brown fox jumps over the lazy dog and then")
    cache = ContiguousKVCache.for_model(model, max_seq_len=64)
    mine = forward_with_own_cache(model, ids, cache, all_logits=True)
    with torch.no_grad():
        theirs = model(input_ids=ids, use_cache=False).logits
    assert mine.shape == theirs.shape
    assert (mine - theirs).abs().max().item() < 1e-3


def test_incremental_prefill_equals_single_prefill(model, encode):
    """Feeding a prompt in chunks must equal feeding it all at once.

    This is chunked prefill, which phase 3 needs in order to mix a long prompt
    into a batch of decoding requests without stalling them.
    """
    ids = encode("The capital of France is Paris and the capital of Germany is")
    whole = ContiguousKVCache.for_model(model, max_seq_len=64)
    logits_whole = forward_with_own_cache(model, ids, whole)

    chunked = ContiguousKVCache.for_model(model, max_seq_len=64)
    split = ids.shape[1] // 2
    forward_with_own_cache(model, ids[:, :split], chunked)
    logits_chunked = forward_with_own_cache(model, ids[:, split:], chunked)

    assert whole.length == chunked.length == ids.shape[1]
    assert (logits_whole - logits_chunked).abs().max().item() < 1e-3


# --- cache mechanics (no model needed beyond shape info) ---

def test_cache_shape_follows_kv_heads(model):
    cache = ContiguousKVCache.for_model(model, max_seq_len=128)
    n_layers, n_kv_heads, seq, head_dim = cache.k.shape
    assert (n_layers, n_kv_heads, seq, head_dim) == (24, 2, 128, 64)
    assert cache.v.shape == cache.k.shape


def test_length_advances_once_per_pass_not_per_layer(model, encode):
    """All 24 layers write at the same offset; only commit() moves the length."""
    ids = encode("hello world")
    cache = ContiguousKVCache.for_model(model, max_seq_len=64)
    forward_with_own_cache(model, ids, cache)
    assert cache.length == ids.shape[1]
    forward_with_own_cache(model, ids[:, :1], cache)
    assert cache.length == ids.shape[1] + 1


def test_utilization_reports_reservation_waste(model, encode):
    """The number that justifies phase 2.

    A short request holding a long reservation should report low utilization.
    """
    ids = encode("The capital of France is")
    result = greedy_own_cache(model, ids, max_new_tokens=8, max_seq_len=512)
    used = len(ids[0]) + len(result.new_token_ids)
    assert result.cache_utilization == pytest.approx(used / 512)
    assert result.cache_utilization < 0.05, "expected most of the reservation wasted"


def test_overflow_raises_rather_than_corrupting(model, encode):
    ids = encode("The capital of France is")
    tiny = ContiguousKVCache.for_model(model, max_seq_len=ids.shape[1])
    forward_with_own_cache(model, ids, tiny)  # exactly fills it
    assert tiny.free_slots() == 0
    with pytest.raises(KVCacheFull):
        forward_with_own_cache(model, ids[:, :1], tiny)


def test_reset_allows_reuse_without_stale_reads(model, encode):
    """A reused buffer must behave like a fresh one even though old data remains.

    reset() deliberately does not zero the tensor. If the second run matches a
    fresh run, nothing past `length` is being read.
    """
    ids = encode(PROMPTS[0])
    shared = ContiguousKVCache.for_model(model, max_seq_len=128)

    first = greedy_own_cache(model, ids, max_new_tokens=12, cache=shared)
    shared.reset()
    assert shared.length == 0
    second = greedy_own_cache(model, ids, max_new_tokens=12, cache=shared)
    fresh = greedy_own_cache(model, ids, max_new_tokens=12)

    assert first.new_token_ids == second.new_token_ids == fresh.new_token_ids


# --- the pieces, unit tested ---

def test_repeat_kv_duplicates_within_groups():
    # 2 KV heads, 3 query heads each -> head order must be [0,0,0,1,1,1]
    x = torch.arange(2 * 4).float().reshape(1, 2, 4, 1)
    out = repeat_kv(x, 3)
    assert out.shape == (1, 6, 4, 1)
    for group in range(2):
        for rep in range(3):
            assert torch.equal(out[0, group * 3 + rep], x[0, group])


def test_repeat_kv_identity_when_no_grouping():
    x = torch.randn(1, 2, 4, 8)
    assert torch.equal(repeat_kv(x, 1), x)


def test_causal_mask_prefill_is_lower_triangular():
    mask = causal_mask(4, 4, torch.float32, torch.device("cpu"))
    blocked = mask[0, 0] < 0
    for q in range(4):
        for k in range(4):
            assert blocked[q, k].item() == (k > q), f"q={q} k={k}"


def test_causal_mask_with_history_offsets_queries():
    """3 new queries on top of 5 cached tokens: query i sits at position 5+i."""
    mask = causal_mask(3, 8, torch.float32, torch.device("cpu"))
    assert mask.shape == (1, 1, 3, 8)
    blocked = mask[0, 0] < 0
    for q in range(3):
        for k in range(8):
            assert blocked[q, k].item() == (k > 5 + q), f"q={q} k={k}"


def test_causal_mask_skipped_for_single_decode_token():
    assert causal_mask(1, 99, torch.float32, torch.device("cpu")) is None
