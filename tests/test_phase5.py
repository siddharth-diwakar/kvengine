"""Phase 5: speculative decoding must be exactly greedy decoding, only faster.

The draft model influences speed alone. That makes the whole phase testable with
one small model and no 3B download:

  self-draft      draft == target, so every proposal is what the target would have
                  chosen. Acceptance should be ~100% and output unchanged.
  hostile draft   a wrapper whose predictions are deliberately wrong. Acceptance
                  collapses to ~0 and output is STILL unchanged. This is the real
                  guarantee: a bad draft costs time, never correctness.

If both hold for every k, the algorithm is right regardless of which models get
plugged in.
"""

import pytest
import torch

from kvengine import (
    ContiguousKVCache,
    PagedKVCache,
    greedy_own_cache,
    hf_greedy,
    speculative_greedy,
)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "Q: What is 17 * 3?\nA:",
]


class HostileDraft:
    """Same network, deliberately wrong predictions.

    Wraps the real model and rolls the vocabulary axis of the final logits, so the
    argmax lands on a token the target will essentially never agree with. Only
    `.model` and `.lm_head` are needed, because that is all the forward pass uses.
    """

    def __init__(self, base):
        self.model = base.model
        self.config = base.config
        self.generation_config = base.generation_config
        self._head = base.lm_head

    def lm_head(self, hidden):
        return self._head(hidden).roll(shifts=1, dims=-1)


def make_caches(model, max_seq_len=256):
    return (
        ContiguousKVCache.for_model(model, max_seq_len=max_seq_len),
        ContiguousKVCache.for_model(model, max_seq_len=max_seq_len),
    )


# --- the anchor: identical to greedy, for any draft, for any k ---

@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_self_draft_matches_greedy_for_every_k(model, encode, k):
    for prompt in PROMPTS:
        ids = encode(prompt)
        t_cache, d_cache = make_caches(model)
        spec = speculative_greedy(
            model, model, ids, t_cache, d_cache, k=k, max_new_tokens=24
        )
        ref = hf_greedy(model, ids, max_new_tokens=24)
        assert spec.new_token_ids == ref.new_token_ids, f"k={k}, prompt={prompt!r}"


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_hostile_draft_still_matches_greedy(model, encode, k):
    """The guarantee that matters: a useless draft cannot corrupt the output."""
    draft = HostileDraft(model)
    for prompt in PROMPTS[:3]:
        ids = encode(prompt)
        t_cache, d_cache = make_caches(model)
        spec = speculative_greedy(
            model, draft, ids, t_cache, d_cache, k=k, max_new_tokens=20
        )
        ref = hf_greedy(model, ids, max_new_tokens=20)
        assert spec.new_token_ids == ref.new_token_ids, f"k={k}, prompt={prompt!r}"


def test_self_draft_acceptance_is_near_total(model, encode):
    """draft == target means every proposal should be accepted.

    Asserted as >0.9 rather than ==1.0 because the draft proposes one token per
    forward while the target verifies k in a batched pass, and batched matmuls can
    differ in the last bits. That can flip a genuine near-tie, which is a rounding
    artefact rather than a bug.
    """
    ids = encode(PROMPTS[2])
    t_cache, d_cache = make_caches(model)
    spec = speculative_greedy(model, model, ids, t_cache, d_cache, k=4, max_new_tokens=32)

    assert spec.acceptance_rate > 0.9, f"self-draft acceptance only {spec.acceptance_rate}"
    assert spec.tokens_per_target_forward > 4.0, (
        "with everything accepted each target pass should yield about k+1 tokens, "
        f"got {spec.tokens_per_target_forward}"
    )


def test_hostile_draft_acceptance_is_near_zero(model, encode):
    """Confirms the hostile draft really is hostile, so its test means something."""
    ids = encode(PROMPTS[2])
    draft = HostileDraft(model)
    t_cache, d_cache = make_caches(model)
    spec = speculative_greedy(model, draft, ids, t_cache, d_cache, k=4, max_new_tokens=20)

    assert spec.acceptance_rate < 0.1, f"draft was not hostile: {spec.acceptance_rate}"
    # Every iteration falls back to the target's own choice: exactly 1 token each.
    assert spec.tokens_per_target_forward == pytest.approx(1.0, abs=0.05)


def test_larger_k_needs_fewer_target_passes(model, encode):
    """The mechanism, stated as a measurement: bigger k, fewer serial target steps."""
    ids = encode(PROMPTS[2])
    passes = {}
    for k in (1, 4, 8):
        t_cache, d_cache = make_caches(model)
        spec = speculative_greedy(
            model, model, ids, t_cache, d_cache, k=k, max_new_tokens=32
        )
        passes[k] = spec.target_forwards
    assert passes[8] < passes[4] < passes[1], f"target passes did not fall with k: {passes}"


# --- budgets and stopping ---

def test_never_exceeds_max_new_tokens(model, encode):
    """An iteration emits up to k+1 tokens, so the budget must be trimmed."""
    ids = encode(PROMPTS[2])
    for budget in (1, 3, 7, 10):
        t_cache, d_cache = make_caches(model)
        spec = speculative_greedy(
            model, model, ids, t_cache, d_cache, k=4, max_new_tokens=budget
        )
        assert len(spec.new_token_ids) == budget, f"budget {budget} not respected"
        ref = hf_greedy(model, ids, max_new_tokens=budget)
        assert spec.new_token_ids == ref.new_token_ids


def test_stops_at_eos_without_emitting_past_it(model, encode):
    """Tokens speculated beyond EOS must be discarded, matching .generate()."""
    ids = encode(PROMPTS[3])  # answers " 51" then EOS
    t_cache, d_cache = make_caches(model)
    spec = speculative_greedy(model, model, ids, t_cache, d_cache, k=4, max_new_tokens=32)
    ref = hf_greedy(model, ids, max_new_tokens=32)

    assert spec.stopped_on_eos
    assert spec.new_token_ids == ref.new_token_ids
    eos = spec.new_token_ids[-1]
    assert spec.new_token_ids.count(eos) == 1, "emitted tokens after EOS"


# --- works on the paged cache too ---

def test_works_with_paged_cache(model, encode):
    pool = PagedKVCache.for_model(model, num_blocks=256, block_size=16)
    ids = encode(PROMPTS[0])
    spec = speculative_greedy(
        model,
        model,
        ids,
        pool.new_sequence("target"),
        pool.new_sequence("draft"),
        k=4,
        max_new_tokens=24,
    )
    ref = hf_greedy(model, ids, max_new_tokens=24)
    assert spec.new_token_ids == ref.new_token_ids


def test_rejected_blocks_return_to_the_pool(model, encode):
    """Rollback on a paged cache should hand blocks back, not just move a counter.

    A hostile draft rejects almost everything, so the target repeatedly writes k
    tokens and rolls back to 1. Block count must track the accepted length.
    """
    pool = PagedKVCache.for_model(model, num_blocks=64, block_size=4)
    ids = encode(PROMPTS[0])
    t_seq = pool.new_sequence("target")
    d_seq = pool.new_sequence("draft")

    spec = speculative_greedy(
        model, HostileDraft(model), ids, t_seq, d_seq, k=8, max_new_tokens=16
    )
    accepted_len = len(ids[0]) + len(spec.new_token_ids)
    # The cache holds accepted_len - 1 tokens; blocks must match that, not the
    # high-water mark reached while speculating 8 tokens ahead.
    assert len(t_seq.blocks) == pool.manager.blocks_needed(t_seq.length)
    assert t_seq.length <= accepted_len
    pool.manager.check_invariants()

    t_seq.free()
    d_seq.free()
    assert pool.manager.num_allocated == 0


# --- truncate, unit tested ---

def test_contiguous_truncate_rolls_back_length(model, encode):
    ids = encode(PROMPTS[0])
    cache = ContiguousKVCache.for_model(model, max_seq_len=64)
    greedy_own_cache(model, ids, max_new_tokens=6, cache=cache)
    before = cache.length

    cache.truncate(3)
    assert cache.length == 3
    with pytest.raises(ValueError):
        cache.truncate(before + 1)  # cannot truncate upward
    with pytest.raises(ValueError):
        cache.truncate(-1)


def test_paged_truncate_frees_whole_blocks_only(model):
    pool = PagedKVCache.for_model(model, num_blocks=32, block_size=4)
    seq = pool.new_sequence("t")
    seq.reserve(16)
    seq.commit(16)
    assert len(seq.blocks) == 4

    seq.truncate(9)  # 9 tokens still needs 3 blocks
    assert seq.length == 9
    assert len(seq.blocks) == 3
    assert pool.manager.num_free == 32 - 3

    seq.truncate(8)  # exactly 2 blocks
    assert len(seq.blocks) == 2

    seq.truncate(0)
    assert len(seq.blocks) == 0
    pool.manager.check_invariants()


def test_truncate_then_continue_matches_uninterrupted_run(model, encode):
    """Rolling back and regenerating must give the same tokens as never rolling back.

    Proves truncate() leaves no stale K/V behind, which is exactly the failure mode
    that would make speculative decoding subtly wrong.
    """
    ids = encode(PROMPTS[0])
    straight = greedy_own_cache(model, ids, max_new_tokens=12, eos_token_id=None)

    from kvengine import forward_with_own_cache

    cache = ContiguousKVCache.for_model(model, max_seq_len=64)
    logits = forward_with_own_cache(model, ids, cache)
    produced = []
    for step in range(12):
        tok = int(torch.argmax(logits[0, -1, :]))
        produced.append(tok)
        # Halfway through, speculate three junk tokens and roll them back.
        if step == 5:
            mark = cache.length
            junk = torch.tensor([[tok, tok, tok]], device=ids.device)
            forward_with_own_cache(model, junk, cache)
            cache.truncate(mark)
        logits = forward_with_own_cache(
            model, torch.tensor([[tok]], device=ids.device), cache
        )
    assert produced == straight.new_token_ids
