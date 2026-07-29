"""Phase 0 correctness anchor.

Everything later in this project is judged against this: a hand-written greedy
loop must produce byte-identical output to HuggingFace's own greedy generation.
"""

import torch

from kvengine import greedy_no_cache, greedy_with_cache, hf_greedy, model_shape_info

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "Q: What is 17 * 3?\nA:",
]


def test_shape_info_uses_grouped_query_attention(model):
    info = model_shape_info(model)
    # Qwen2.5-0.5B: 24 layers, 14 query heads, 2 KV heads, head_dim 64.
    # The cache is sized by KV heads; asserting they differ keeps a future
    # refactor from silently sizing it off num_attention_heads.
    assert info["num_key_value_heads"] < info["num_attention_heads"]
    assert info["num_layers"] > 0
    assert info["head_dim"] > 0


def test_cached_loop_matches_hf_generate(model, encode):
    for prompt in PROMPTS:
        ids = encode(prompt)
        mine = greedy_with_cache(model, ids, max_new_tokens=24)
        ref = hf_greedy(model, ids, max_new_tokens=24)
        assert mine.new_token_ids == ref.new_token_ids, f"diverged on prompt: {prompt!r}"


def test_uncached_loop_matches_cached_loop(model, encode):
    """If these disagree, the cache is returning stale or misaligned history."""
    for prompt in PROMPTS[:2]:
        ids = encode(prompt)
        cached = greedy_with_cache(model, ids, max_new_tokens=16)
        uncached = greedy_no_cache(model, ids, max_new_tokens=16)
        assert cached.new_token_ids == uncached.new_token_ids, f"prompt: {prompt!r}"


def test_cached_and_uncached_logits_agree_numerically(model, encode):
    """Stronger than matching token ids: the actual scores should line up.

    Token ids can match by luck when the top choice wins by a wide margin. This
    catches a cache that is subtly wrong but not yet wrong enough to change the
    argmax.
    """
    ids = encode(PROMPTS[0])
    # eos_token_id=None disables early stopping so both paths run all 8 steps
    # and the logit tensors are directly comparable.
    cached = greedy_with_cache(model, ids, max_new_tokens=8, eos_token_id=None, collect_logits=True)
    uncached = greedy_no_cache(model, ids, max_new_tokens=8, eos_token_id=None, collect_logits=True)
    max_diff = (cached.step_logits - uncached.step_logits).abs().max().item()
    assert max_diff < 1e-3, f"logits drift between cached and uncached paths: {max_diff}"


def test_eos_halts_generation(model, encode):
    """Stop on EOS, and include the EOS token, exactly as .generate() does."""
    ids = encode(PROMPTS[0])
    # Discover what the model would emit first, then declare that token EOS.
    # Gives a deterministic stop without needing a prompt that runs to a real one.
    first = greedy_with_cache(model, ids, max_new_tokens=1).new_token_ids[0]

    mine = greedy_with_cache(model, ids, max_new_tokens=24, eos_token_id=first)
    assert mine.new_token_ids == [first]
    assert mine.stopped_on_eos

    ref = hf_greedy(model, ids, max_new_tokens=24, eos_token_id=first)
    assert mine.new_token_ids == ref.new_token_ids


def test_prompt_is_not_mutated(model, encode):
    """The loops must not modify the caller's input tensor in place."""
    ids = encode(PROMPTS[0])
    before = ids.clone()
    greedy_with_cache(model, ids, max_new_tokens=4)
    greedy_no_cache(model, ids, max_new_tokens=4)
    assert torch.equal(ids, before)
