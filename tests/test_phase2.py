"""Phase 2: paged attention must be indistinguishable from contiguous attention.

Same correctness anchor as phases 0 and 1, now with the history scattered across
blocks. The interesting tests are the ones that force the block table to be
non-contiguous, because a gather bug is invisible when blocks happen to be
handed out in order.
"""

import pytest
import torch

from kvengine import (
    OutOfBlocks,
    PagedKVCache,
    forward_with_own_cache,
    greedy_own_cache,
    greedy_paged,
    hf_greedy,
)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "Q: What is 17 * 3?\nA:",
]


# --- the anchor ---

def test_paged_matches_hf_generate(model, encode):
    pool = PagedKVCache.for_model(model, num_blocks=256, block_size=16)
    for prompt in PROMPTS:
        ids = encode(prompt)
        mine = greedy_paged(model, ids, pool, max_new_tokens=24)
        ref = hf_greedy(model, ids, max_new_tokens=24)
        assert mine.new_token_ids == ref.new_token_ids, f"diverged on prompt: {prompt!r}"
    pool.manager.check_invariants()
    assert pool.manager.num_allocated == 0, "blocks leaked after requests retired"


@pytest.mark.parametrize("block_size", [1, 2, 3, 7, 16, 17, 64])
def test_paged_matches_across_block_sizes(model, encode, block_size):
    """Block size must not change a single token.

    block_size=1 makes every token its own block (maximum scatter), and the
    primes catch arithmetic that only works when the size divides the sequence
    length evenly.
    """
    pool = PagedKVCache.for_model(model, num_blocks=512, block_size=block_size)
    ids = encode(PROMPTS[0])
    mine = greedy_paged(model, ids, pool, max_new_tokens=16)
    ref = hf_greedy(model, ids, max_new_tokens=16)
    assert mine.new_token_ids == ref.new_token_ids


def test_paged_logits_match_contiguous_logits(model, encode):
    """Compare scores, not just tokens, against the phase 1 cache."""
    pool = PagedKVCache.for_model(model, num_blocks=128, block_size=16)
    ids = encode(PROMPTS[0])
    paged = greedy_paged(model, ids, pool, max_new_tokens=8, eos_token_id=None, collect_logits=True)
    contiguous = greedy_own_cache(model, ids, max_new_tokens=8, eos_token_id=None, collect_logits=True)
    max_diff = (paged.step_logits - contiguous.step_logits).abs().max().item()
    assert max_diff < 1e-3, f"paged gather drifts from contiguous: {max_diff}"


def test_prompt_landing_exactly_on_block_boundary(model, tokenizer, encode):
    """Off-by-one in blocks_needed shows up only when length % block_size == 0."""
    pool = PagedKVCache.for_model(model, num_blocks=64, block_size=4)
    ids = encode(PROMPTS[0])
    trimmed = ids[:, : (ids.shape[1] // 4) * 4]  # trim to an exact multiple
    assert trimmed.shape[1] % 4 == 0 and trimmed.shape[1] > 0

    result = greedy_paged(model, trimmed, pool, max_new_tokens=9, free_on_finish=False)
    ref = hf_greedy(model, trimmed, max_new_tokens=9)
    assert result.new_token_ids == ref.new_token_ids
    result.sequence.free()
    pool.manager.check_invariants()


def test_matches_when_block_table_is_fragmented(model, encode):
    """The test that actually exercises the gather.

    Blocks handed out in ascending order would make a broken gather look correct,
    so the pool is deliberately fragmented first and the block table is asserted
    to be out of order before the output is checked.
    """
    pool = PagedKVCache.for_model(model, num_blocks=64, block_size=4)

    # Fill the low blocks, then punch holes so the next request gets scraps.
    hogs = [pool.new_sequence(f"hog{i}") for i in range(16)]
    for h in hogs:
        h.reserve(4)
        h.commit(4)
    for h in hogs[::2]:
        h.free()

    ids = encode(PROMPTS[0])
    result = greedy_paged(model, ids, pool, max_new_tokens=16, free_on_finish=False)
    blocks = result.sequence.blocks

    assert len(blocks) > 2, "need several blocks for this test to mean anything"
    assert any(blocks[i + 1] != blocks[i] + 1 for i in range(len(blocks) - 1)), (
        f"block table came out contiguous ({blocks}); gather was not exercised"
    )

    ref = hf_greedy(model, ids, max_new_tokens=16)
    assert result.new_token_ids == ref.new_token_ids

    result.sequence.free()
    for h in hogs[1::2]:
        h.free()
    pool.manager.check_invariants()
    assert pool.manager.num_allocated == 0


def test_two_interleaved_sequences_do_not_corrupt_each_other(model, encode):
    """Step two requests alternately through the same pool, token by token.

    This is the phase 3 setup arriving early: if writes ever landed in another
    request's blocks, interleaved output would differ from running each alone.
    """
    pool = PagedKVCache.for_model(model, num_blocks=128, block_size=4)
    ids_a = encode(PROMPTS[0])
    ids_b = encode(PROMPTS[2])

    seq_a = pool.new_sequence("a")
    seq_b = pool.new_sequence("b")

    def prefill(ids, seq):
        logits = forward_with_own_cache(model, ids, seq)
        return int(torch.argmax(logits[0, -1, :]))

    tok_a = prefill(ids_a, seq_a)
    tok_b = prefill(ids_b, seq_b)
    out_a, out_b = [tok_a], [tok_b]

    for _ in range(15):
        for seq, out in ((seq_a, out_a), (seq_b, out_b)):
            nxt = torch.tensor([[out[-1]]], device=ids_a.device)
            logits = forward_with_own_cache(model, nxt, seq)
            out.append(int(torch.argmax(logits[0, -1, :])))
        assert not set(seq_a.blocks) & set(seq_b.blocks), "block tables overlap"

    alone_a = hf_greedy(model, ids_a, max_new_tokens=16, eos_token_id=None)
    alone_b = hf_greedy(model, ids_b, max_new_tokens=16, eos_token_id=None)
    assert out_a == alone_a.new_token_ids, "request A corrupted by interleaving"
    assert out_b == alone_b.new_token_ids, "request B corrupted by interleaving"

    seq_a.free()
    seq_b.free()
    pool.manager.check_invariants()


# --- allocation behaviour under the model ---

def test_blocks_are_allocated_lazily_as_the_sequence_grows(model, encode):
    pool = PagedKVCache.for_model(model, num_blocks=64, block_size=4)
    ids = encode(PROMPTS[0])
    seq = pool.new_sequence("grow")

    forward_with_own_cache(model, ids, seq)
    after_prefill = len(seq.blocks)
    assert after_prefill == pool.manager.blocks_needed(ids.shape[1])

    # Decoding within the current block must not allocate.
    while seq.free_slots() > 0:
        before = len(seq.blocks)
        forward_with_own_cache(model, ids[:, :1], seq)
        assert len(seq.blocks) == before, "allocated a block while one had room"

    # The next token crosses the boundary and must allocate exactly one block.
    forward_with_own_cache(model, ids[:, :1], seq)
    assert len(seq.blocks) == after_prefill + 1

    seq.free()
    pool.manager.check_invariants()


def test_utilization_is_bounded_by_block_size(model, encode):
    """The payoff number: waste is capped at block_size - 1 slots per request.

    Compare against phase 1, where a short request holding a 2048-slot
    reservation reported under 3% utilization.
    """
    pool = PagedKVCache.for_model(model, num_blocks=256, block_size=16)
    ids = encode(PROMPTS[0])
    result = greedy_paged(model, ids, pool, max_new_tokens=8, free_on_finish=False)

    used = ids.shape[1] + len(result.new_token_ids)
    held = result.sequence.capacity
    assert held - used < 16, "waste exceeded one block"
    assert result.cache_utilization > 0.7, f"utilization unexpectedly low: {result.cache_utilization}"
    result.sequence.free()


def test_pool_exhaustion_raises_and_leaves_pool_consistent(model, encode):
    """Running out of blocks must be a clean rejection, not corruption."""
    ids = encode(PROMPTS[2])
    pool = PagedKVCache.for_model(model, num_blocks=2, block_size=4)  # 8 slots total

    with pytest.raises(OutOfBlocks):
        greedy_paged(model, ids, pool, max_new_tokens=32)

    # greedy_paged frees on the way out even when the forward pass raised.
    pool.manager.check_invariants()
    assert pool.manager.num_allocated == 0, "failed request leaked its blocks"
    assert pool.manager.num_free == 2

    # And the pool still serves a request that fits.
    short = ids[:, :4]
    result = greedy_paged(model, short, pool, max_new_tokens=3)
    assert len(result.new_token_ids) == 3
    pool.manager.check_invariants()


def test_sequential_requests_recycle_blocks(model, encode):
    """Many requests through a small pool: no leaks, identical output every time."""
    pool = PagedKVCache.for_model(model, num_blocks=16, block_size=8)
    ids = encode(PROMPTS[0])
    baseline = None

    for i in range(8):
        result = greedy_paged(model, ids, pool, max_new_tokens=12, request_id=f"r{i}")
        if baseline is None:
            baseline = result.new_token_ids
        assert result.new_token_ids == baseline, f"run {i} differed after recycling"
        assert pool.manager.num_allocated == 0
        pool.manager.check_invariants()


def test_recycled_blocks_carry_no_stale_data(model, encode):
    """A block reused by a second request must not leak the first request's keys.

    Blocks are never zeroed on free, on purpose: if a stale read existed, this
    test would catch it, whereas zeroing would hide it.
    """
    pool = PagedKVCache.for_model(model, num_blocks=8, block_size=8)

    long_first = greedy_paged(model, encode(PROMPTS[2]), pool, max_new_tokens=20)
    assert pool.manager.num_allocated == 0

    ids = encode(PROMPTS[0])
    after = greedy_paged(model, ids, pool, max_new_tokens=12)
    fresh_pool = PagedKVCache.for_model(model, num_blocks=8, block_size=8)
    fresh = greedy_paged(model, ids, fresh_pool, max_new_tokens=12)

    assert after.new_token_ids == fresh.new_token_ids, "stale block data leaked through"
    assert long_first.new_token_ids  # sanity: the first request really ran


# --- the gather, unit tested directly ---

def test_write_then_gather_roundtrips_known_values(model):
    """Scatter/gather in isolation, with recognisable numbers instead of weights."""
    pool = PagedKVCache.for_model(model, num_blocks=8, block_size=4)
    seq = pool.new_sequence("rt")
    n_heads, head_dim = pool.num_kv_heads, pool.head_dim

    # Token t gets the constant value t across every head and channel.
    n = 10
    k = torch.arange(1, n + 1, dtype=pool.dtype, device=pool.device)
    k = k.view(1, 1, n, 1).expand(1, n_heads, n, head_dim).contiguous()
    v = k * 100

    seq.reserve(n)
    k_all, v_all = seq.write(0, k, v, start=0)
    seq.commit(n)

    assert k_all.shape == (1, n_heads, n, head_dim)
    assert torch.equal(k_all, k), "gathered keys do not match what was written"
    assert torch.equal(v_all, v)
    assert len(seq.blocks) == 3  # 10 tokens over blocks of 4
    seq.free()


def test_gather_spans_blocks_in_logical_order(model):
    """Blocks must be stitched in block-table order, not physical block order.

    Forcing a descending block table catches a gather that implicitly assumes
    ascending physical ids.
    """
    pool = PagedKVCache.for_model(model, num_blocks=8, block_size=2)
    n_heads, head_dim = pool.num_kv_heads, pool.head_dim

    # Drain and release so the LIFO free list hands blocks back descending.
    drained = pool.manager.allocate(8, owner="drain")
    pool.manager.free(drained)

    seq = pool.new_sequence("order")
    n = 6
    k = torch.arange(1, n + 1, dtype=pool.dtype, device=pool.device)
    k = k.view(1, 1, n, 1).expand(1, n_heads, n, head_dim).contiguous()

    # Write one token at a time so blocks are acquired incrementally.
    for t in range(n):
        seq.reserve(1)
        seq.write(0, k[:, :, t : t + 1, :], k[:, :, t : t + 1, :], start=t)
        seq.commit(1)

    k_all, _ = seq._gather(0, n)
    assert torch.equal(k_all, k), (
        f"gather returned wrong order for block table {seq.blocks}"
    )
    seq.free()
