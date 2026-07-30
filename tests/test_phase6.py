"""Phase 6: prefix sharing must save work and change nothing.

Two halves. The allocator tests are pure logic and run in milliseconds; the engine
tests prove that sharing blocks between requests leaves output byte-identical while
measurably reducing prefill.

The tests that matter most are the ones asserting sharing *actually happened*.
"Output unchanged" passes trivially if nothing was ever shared, so every
correctness test here is paired with a check that the prefix cache was hit.
"""

import pytest

from kvengine import (
    BlockManager,
    DoubleFree,
    Engine,
    OutOfBlocks,
    PagedKVCache,
    hf_greedy,
)
from kvengine.blocks import ROOT_DIGEST, chain_digest

# Long enough to span several 16-token blocks, so there is a real prefix to share.
SYSTEM = (
    "You are a careful and concise assistant. Answer directly, explain your "
    "reasoning in one sentence, and never speculate beyond the evidence given. "
    "If you are unsure about something, say so plainly instead of guessing. "
)
QUESTIONS = [
    "What is the capital of France?",
    "Name three prime numbers.",
    "Why is the sky blue?",
    "What does a compiler do?",
]


# ---------------------------------------------------------------- allocator

def test_incref_shares_a_block_without_reallocating():
    m = BlockManager(8, 4)
    blocks = m.allocate(2, owner="a")
    m.incref(blocks)

    assert m.refcount(blocks[0]) == 2
    assert m.num_allocated == 2, "sharing should not consume extra blocks"
    assert m.num_free == 6
    m.check_invariants()


def test_block_returns_to_pool_only_at_zero_references():
    m = BlockManager(4, 4)
    blocks = m.allocate(1, owner="a")
    m.incref(blocks)

    m.free(blocks)
    assert m.num_allocated == 1, "released too early, while still shared"
    assert m.refcount(blocks[0]) == 1

    m.free(blocks)
    assert m.num_allocated == 0
    assert m.num_free == 4
    m.check_invariants()


def test_free_below_zero_is_rejected():
    m = BlockManager(4, 4)
    blocks = m.allocate(1, owner="a")
    m.free(blocks)
    with pytest.raises(DoubleFree):
        m.free(blocks)
    m.check_invariants()


def test_registered_block_becomes_reclaimable_not_free():
    """A cached prefix outlives the request that computed it.

    This is what makes prefix caching pay off for *sequential* requests, not just
    concurrent ones.
    """
    m = BlockManager(8, 4)
    (block,) = m.allocate(1, owner="a")
    digest = chain_digest(ROOT_DIGEST, [1, 2, 3, 4])
    assert m.register_prefix(block, digest, [1, 2, 3, 4])

    m.free([block])
    assert m.num_allocated == 0
    assert m.num_reclaimable == 1
    assert m.num_unused == 7
    assert m.num_free == 8, "reclaimable blocks are still allocatable"
    m.check_invariants()

    # And it can still be found and resurrected.
    assert m.lookup_prefix(digest, [1, 2, 3, 4]) == block
    m.incref([block])
    assert m.refcount(block) == 1
    assert m.num_reclaimable == 0
    m.check_invariants()


def test_allocate_spends_unused_blocks_before_cached_ones():
    """Only discard a cached prefix when the space is genuinely needed."""
    m = BlockManager(4, 4)
    (cached,) = m.allocate(1, owner="a")
    m.register_prefix(cached, chain_digest(ROOT_DIGEST, [9, 9, 9, 9]), [9, 9, 9, 9])
    m.free([cached])
    assert m.num_reclaimable == 1

    taken = m.allocate(3, owner="b")
    assert cached not in taken, "evicted a cached block while unused ones remained"
    assert m.num_reclaimable == 1
    assert m.evictions == 0
    m.check_invariants()


def test_eviction_is_lru_and_unregisters():
    m = BlockManager(3, 4)
    ids = m.allocate(3, owner="a")
    digests = []
    for i, b in enumerate(ids):
        toks = [i, i, i, i]
        d = chain_digest(ROOT_DIGEST, toks)
        m.register_prefix(b, d, toks)
        digests.append((d, toks))
    m.free(ids)  # all three become reclaimable, in order ids[0], ids[1], ids[2]
    assert m.num_reclaimable == 3

    (taken,) = m.allocate(1, owner="b")
    assert taken == ids[0], "did not evict the least recently released block"
    assert m.evictions == 1
    # Its registry entry is gone, so nobody can adopt stale storage.
    assert m.lookup_prefix(*digests[0]) is None
    # The others survive.
    assert m.lookup_prefix(*digests[1]) == ids[1]
    m.check_invariants()


def test_lookup_verifies_tokens_not_just_the_digest():
    """Guards against a digest collision silently serving the wrong keys/values."""
    m = BlockManager(4, 4)
    (block,) = m.allocate(1, owner="a")
    digest = chain_digest(ROOT_DIGEST, [1, 2, 3, 4])
    m.register_prefix(block, digest, [1, 2, 3, 4])

    assert m.lookup_prefix(digest, [1, 2, 3, 4]) == block
    assert m.lookup_prefix(digest, [1, 2, 3, 5]) is None, "served a block on token mismatch"


def test_partial_blocks_cannot_be_registered():
    """The rule that removes the need for copy-on-write entirely."""
    m = BlockManager(4, 4)
    (block,) = m.allocate(1, owner="a")
    with pytest.raises(ValueError):
        m.register_prefix(block, chain_digest(ROOT_DIGEST, [1, 2]), [1, 2])


def test_digest_chain_binds_a_block_to_its_position():
    """The same tokens at a different offset must not match.

    Keys and values have their absolute position baked in by rotary embeddings, so a
    block cached for positions 0-3 is not reusable at positions 4-7. Chaining the
    digest through every preceding token enforces that automatically.
    """
    toks = [7, 7, 7, 7]
    at_start = chain_digest(ROOT_DIGEST, toks)
    after_something = chain_digest(chain_digest(ROOT_DIGEST, [1, 2, 3, 4]), toks)
    assert at_start != after_something


def test_unregister_frees_an_unreferenced_block():
    m = BlockManager(4, 4)
    (block,) = m.allocate(1, owner="a")
    digest = chain_digest(ROOT_DIGEST, [1, 2, 3, 4])
    m.register_prefix(block, digest, [1, 2, 3, 4])
    m.free([block])
    assert m.num_reclaimable == 1

    assert m.unregister_prefix(block)
    assert m.num_reclaimable == 0
    assert m.num_unused == 4
    assert m.lookup_prefix(digest, [1, 2, 3, 4]) is None
    m.check_invariants()


def test_sharing_churn_preserves_invariants():
    """Randomised share/release churn, invariants checked after every operation."""
    import random

    rng = random.Random(99)
    m = BlockManager(32, 4)
    live: dict[str, list[int]] = {}
    shared_pool: list[int] = []
    next_id = 0

    for _ in range(2000):
        roll = rng.random()
        if live and roll < 0.35:
            owner = rng.choice(list(live))
            m.free(live.pop(owner))
        elif shared_pool and roll < 0.55:
            # A new request adopts existing blocks instead of allocating.
            owner = f"s{next_id}"; next_id += 1
            take = rng.sample(shared_pool, min(len(shared_pool), rng.randint(1, 3)))
            take = [b for b in take if m.refcount(b) > 0]
            if take:
                m.incref(take)
                live[owner] = take
        else:
            owner = f"r{next_id}"; next_id += 1
            try:
                blocks = m.allocate(rng.randint(1, 5), owner=owner)
                live[owner] = blocks
                shared_pool = [b for b in (shared_pool + blocks) if m.refcount(b) > 0][-12:]
            except OutOfBlocks:
                pass
        m.check_invariants()

    for blocks in live.values():
        m.free(blocks)
    m.check_invariants()
    assert m.num_allocated == 0, "references leaked over the run"


# ---------------------------------------------------------------- engine

def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def run(model, encode, prompts, max_new_tokens=16, share=True, blocks=256, block_size=16):
    pool = PagedKVCache.for_model(model, num_blocks=blocks, block_size=block_size)
    engine = Engine(model, pool, max_batch_size=len(prompts), share_prefixes=share)
    for i, p in enumerate(prompts):
        engine.add_request(encode(p)[0].tolist(), max_new_tokens=max_new_tokens,
                           request_id=f"r{i}")
    finished = engine.run()
    return engine, {r.request_id: r for r in finished}


def test_prompts_actually_share_a_prefix(encode):
    """Establishes the premise the rest of these tests depend on."""
    toks = [encode(SYSTEM + q)[0].tolist() for q in QUESTIONS]
    for other in toks[1:]:
        assert common_prefix_len(toks[0], other) >= 16, (
            "prompts do not share a full block; the sharing tests would be vacuous"
        )


def test_sharing_changes_no_output(model, encode):
    prompts = [SYSTEM + q for q in QUESTIONS]
    ref = {p: hf_greedy(model, encode(p), max_new_tokens=16).new_token_ids for p in prompts}

    shared_engine, shared = run(model, encode, prompts, share=True)
    _, private = run(model, encode, prompts, share=False)

    for i, p in enumerate(prompts):
        rid = f"r{i}"
        assert shared[rid].output_token_ids == ref[p], f"sharing changed output: {p!r}"
        assert shared[rid].output_token_ids == private[rid].output_token_ids

    # ...and prove sharing actually occurred, or the above is vacuous.
    assert shared_engine.stats()["prefix_tokens_reused"] > 0, "no prefix was ever reused"


def test_sharing_reduces_prefill_work(model, encode):
    prompts = [SYSTEM + q for q in QUESTIONS]
    shared_engine, _ = run(model, encode, prompts, share=True)
    private_engine, _ = run(model, encode, prompts, share=False)

    shared = shared_engine.stats()
    private = private_engine.stats()
    assert shared["prefill_tokens"] < private["prefill_tokens"], (
        f"prefill not reduced: {shared['prefill_tokens']} vs {private['prefill_tokens']}"
    )
    assert shared["prefix_tokens_reused"] > 0
    assert private["prefix_tokens_reused"] == 0


def test_shared_blocks_are_physically_the_same_blocks(model, encode):
    """Not just fewer tokens prefilled: the same storage, refcounted."""
    pool = PagedKVCache.for_model(model, num_blocks=256, block_size=16)
    prompt_a = (SYSTEM + QUESTIONS[0])
    prompt_b = (SYSTEM + QUESTIONS[1])

    seq_a = pool.new_sequence("a")
    toks_a = encode(prompt_a)[0].tolist()
    from kvengine import forward_with_own_cache
    import torch

    seq_a.adopt_prefix(toks_a)
    forward_with_own_cache(model, torch.tensor([toks_a], device=pool.device), seq_a)
    seq_a.register_prefix_blocks(toks_a)

    seq_b = pool.new_sequence("b")
    toks_b = encode(prompt_b)[0].tolist()
    reused = seq_b.adopt_prefix(toks_b)

    assert reused > 0, "second sequence adopted nothing"
    n_shared = reused // 16
    assert seq_b.blocks[:n_shared] == seq_a.blocks[:n_shared], "different storage"
    for b in seq_b.blocks[:n_shared]:
        assert pool.manager.refcount(b) == 2

    seq_a.free()
    seq_b.free()
    pool.manager.check_invariants()
    assert pool.manager.num_allocated == 0


def test_unrelated_prompts_do_not_share(model, encode):
    prompts = ["The capital of France is", "def fibonacci(n):"]
    engine, _ = run(model, encode, prompts, share=True)
    assert engine.stats()["prefix_tokens_reused"] == 0, "shared blocks between unrelated prompts"


def test_cached_prefix_survives_the_request_that_made_it(model, encode):
    """Sequential requests, not concurrent: the second run should hit the cache."""
    pool = PagedKVCache.for_model(model, num_blocks=256, block_size=16)
    engine = Engine(model, pool, max_batch_size=1, share_prefixes=True)

    engine.add_request(encode(SYSTEM + QUESTIONS[0])[0].tolist(), max_new_tokens=8, request_id="first")
    engine.run()
    assert engine.stats()["prefix_tokens_reused"] == 0  # nothing cached yet
    assert pool.manager.num_reclaimable > 0, "prefix blocks were dropped on retire"

    engine.add_request(encode(SYSTEM + QUESTIONS[1])[0].tolist(), max_new_tokens=8, request_id="second")
    engine.run()
    second = next(r for r in engine.finished if r.request_id == "second")
    assert second.prefix_tokens_reused > 0, "sequential request missed the cached prefix"
    pool.manager.check_invariants()


def test_correct_under_eviction_pressure(model, encode):
    """A pool too small to keep every prefix must still produce right answers."""
    prompts = [SYSTEM + q for q in QUESTIONS]
    ref = {p: hf_greedy(model, encode(p), max_new_tokens=12).new_token_ids for p in prompts}

    engine, results = run(model, encode, prompts, max_new_tokens=12,
                          share=True, blocks=22, block_size=16)
    for i, p in enumerate(prompts):
        assert results[f"r{i}"].output_token_ids == ref[p], f"wrong under eviction: {p!r}"

    engine.pool.manager.check_invariants()
    assert engine.pool.manager.num_allocated == 0, "references leaked"


def test_no_reference_leak_after_many_shared_requests(model, encode):
    pool = PagedKVCache.for_model(model, num_blocks=128, block_size=16)
    engine = Engine(model, pool, max_batch_size=2, share_prefixes=True)
    for i in range(8):
        engine.add_request(
            encode(SYSTEM + QUESTIONS[i % len(QUESTIONS)])[0].tolist(),
            max_new_tokens=6, request_id=f"r{i}",
        )
    engine.run()
    assert len(engine.finished) == 8
    assert pool.manager.num_allocated == 0, "references leaked across shared requests"
    pool.manager.check_invariants()


def test_rollback_into_a_shared_block_is_refused(model, encode):
    """Combining sharing with speculative rollback must fail loudly, not corrupt.

    Reachable only by publishing blocks whose tokens are not final; the guard exists
    so that mistake cannot silently corrupt another request's history.
    """
    import torch
    from kvengine import forward_with_own_cache

    pool = PagedKVCache.for_model(model, num_blocks=64, block_size=4)
    toks = encode(SYSTEM)[0].tolist()[:16]

    seq_a = pool.new_sequence("a")
    forward_with_own_cache(model, torch.tensor([toks], device=pool.device), seq_a)
    seq_a.register_prefix_blocks(toks)

    seq_b = pool.new_sequence("b")
    assert seq_b.adopt_prefix(toks) > 0

    with pytest.raises(ValueError, match="shared"):
        seq_a.truncate(2)  # would rewrite a block seq_b adopted

    seq_a.free()
    seq_b.free()
    pool.manager.check_invariants()
