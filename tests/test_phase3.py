"""Phase 3: continuous batching must not change a single token.

Batching is a throughput optimisation, so the bar is that it is invisible in the
output. Every test here compares engine output against the same prompt run alone
through HuggingFace, including under memory pressure severe enough to force
preemption and recomputation.
"""

import pytest
import torch

from kvengine import (
    Engine,
    PagedKVCache,
    RequestState,
    forward_decode_batch,
    hf_greedy,
    plan_decode_batch,
)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "Q: What is 17 * 3?\nA:",
    "The three laws of thermodynamics are",
    "import numpy as np\n\ndef softmax(x):",
]


def reference_outputs(model, encode, prompts, max_new_tokens):
    return {
        p: hf_greedy(model, encode(p), max_new_tokens=max_new_tokens).new_token_ids
        for p in prompts
    }


def run_engine(model, encode, prompts, max_new_tokens, **engine_kwargs):
    pool = engine_kwargs.pop("pool", None) or PagedKVCache.for_model(
        model, num_blocks=256, block_size=16
    )
    engine = Engine(model, pool, **engine_kwargs)
    for i, prompt in enumerate(prompts):
        engine.add_request(
            encode(prompt)[0].tolist(), max_new_tokens=max_new_tokens, request_id=f"r{i}"
        )
    finished = engine.run()
    return engine, {prompts[int(r.request_id[1:])]: r for r in finished}


# --- the anchor ---

def test_batched_output_matches_single_request(model, encode):
    ref = reference_outputs(model, encode, PROMPTS, 24)
    engine, results = run_engine(model, encode, PROMPTS, 24, max_batch_size=6)

    for prompt, req in results.items():
        assert req.output_token_ids == ref[prompt], f"diverged: {prompt!r}"
        assert req.state is RequestState.FINISHED

    assert engine.stats()["max_decode_batch"] > 1, "never actually batched"
    engine.pool.manager.check_invariants()
    assert engine.pool.manager.num_allocated == 0, "blocks leaked after all retired"


@pytest.mark.parametrize("max_batch_size", [1, 2, 3, 6])
def test_batch_size_does_not_change_output(model, encode, max_batch_size):
    """Batch size is a throughput knob and must have zero effect on tokens."""
    ref = reference_outputs(model, encode, PROMPTS[:4], 16)
    _, results = run_engine(model, encode, PROMPTS[:4], 16, max_batch_size=max_batch_size)
    for prompt, req in results.items():
        assert req.output_token_ids == ref[prompt], f"batch={max_batch_size}: {prompt!r}"


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_block_size_does_not_change_output(model, encode, block_size):
    ref = reference_outputs(model, encode, PROMPTS[:3], 16)
    pool = PagedKVCache.for_model(model, num_blocks=512, block_size=block_size)
    _, results = run_engine(model, encode, PROMPTS[:3], 16, pool=pool, max_batch_size=3)
    for prompt, req in results.items():
        assert req.output_token_ids == ref[prompt]


def test_ragged_lengths_are_masked_correctly(model, encode):
    """Prompts of very different lengths in one batch exercise the padding mask.

    If padding were not masked, the short request would attend to whatever sits
    in the padded slots -- which is another request's data.
    """
    prompts = ["Hi", PROMPTS[2], "The capital of France is"]
    ref = reference_outputs(model, encode, prompts, 20)
    engine, results = run_engine(model, encode, prompts, 20, max_batch_size=3)

    for prompt, req in results.items():
        assert req.output_token_ids == ref[prompt], f"diverged: {prompt!r}"

    decode_steps = [s for s in engine.history if s.kind == "decode"]
    assert any(s.padding_waste > 0 for s in decode_steps), (
        "lengths were not actually ragged, so the mask went untested"
    )


# --- continuous behaviour ---

def test_requests_retire_and_free_blocks_mid_run(model, encode):
    """A finished request must return its blocks immediately, not at the end.

    That is the difference between continuous batching and static batching: the
    batch does not wait for its slowest member.
    """
    prompts = ["Q: What is 17 * 3?\nA:", PROMPTS[2]]  # first stops early on EOS
    engine, results = run_engine(model, encode, prompts, 40, max_batch_size=2)

    short = results[prompts[0]]
    long = results[prompts[1]]
    assert short.finish_reason == "eos"
    assert short.finish_step < long.finish_step, "short request did not retire early"

    # After the short one retired, blocks in use must have dropped while the long
    # one was still generating.
    at_retire = next(s for s in engine.history if short.request_id in s.finished)
    later = [s for s in engine.history if s.step > at_retire.step and s.kind == "decode"]
    assert later, "no decode steps after the early retirement"
    assert min(s.batch_size for s in later) == 1, "batch did not shrink after retirement"


def test_requests_added_mid_run_are_picked_up(model, encode):
    """Staggered arrivals, which is the actual serving pattern."""
    pool = PagedKVCache.for_model(model, num_blocks=256, block_size=16)
    engine = Engine(model, pool, max_batch_size=4)
    ref = reference_outputs(model, encode, PROMPTS[:3], 16)

    engine.add_request(encode(PROMPTS[0])[0].tolist(), max_new_tokens=16, request_id="r0")
    for _ in range(5):
        engine.step()
    engine.add_request(encode(PROMPTS[1])[0].tolist(), max_new_tokens=16, request_id="r1")
    for _ in range(3):
        engine.step()
    engine.add_request(encode(PROMPTS[2])[0].tolist(), max_new_tokens=16, request_id="r2")
    engine.run()

    by_id = {r.request_id: r for r in engine.finished}
    assert len(by_id) == 3
    for i, prompt in enumerate(PROMPTS[:3]):
        assert by_id[f"r{i}"].output_token_ids == ref[prompt], f"late arrival {i} diverged"
    assert engine.pool.manager.num_allocated == 0


def test_decode_first_policy_matches_prefill_first(model, encode):
    """Scheduling policy is a latency/throughput knob, not a correctness one."""
    ref = reference_outputs(model, encode, PROMPTS[:4], 16)
    _, results = run_engine(
        model, encode, PROMPTS[:4], 16, max_batch_size=4, prefill_first=False
    )
    for prompt, req in results.items():
        assert req.output_token_ids == ref[prompt], f"decode_first diverged: {prompt!r}"


# --- preemption ---

def test_preemption_preserves_output_exactly(model, encode):
    """The hardest correctness test in the project.

    The pool is deliberately too small to hold every request at once, so the
    scheduler must preempt, free blocks, and later rebuild the cache by
    recomputing prompt + tokens generated so far. If the resume boundary is off by
    one token, output diverges.
    """
    prompts = PROMPTS[:4]
    ref = reference_outputs(model, encode, prompts, 24)
    pool = PagedKVCache.for_model(model, num_blocks=14, block_size=4)

    engine, results = run_engine(model, encode, prompts, 24, pool=pool, max_batch_size=4)
    stats = engine.stats()

    assert stats["preemptions"] > 0, "pool was not small enough to force preemption"
    assert stats["aborted"] == 0, "requests were aborted, not preempted"
    for prompt, req in results.items():
        assert req.output_token_ids == ref[prompt], (
            f"preemption corrupted output for {prompt!r} "
            f"(preempted {req.preemptions}x)"
        )
    engine.pool.manager.check_invariants()
    assert engine.pool.manager.num_allocated == 0


def test_recomputation_costs_extra_prefill_tokens(model, encode):
    """Preemption should be visibly paid for, so the benchmark can price it."""
    prompts = PROMPTS[:4]
    pool = PagedKVCache.for_model(model, num_blocks=14, block_size=4)
    engine, results = run_engine(model, encode, prompts, 24, pool=pool, max_batch_size=4)

    preempted = [r for r in results.values() if r.preemptions > 0]
    assert preempted, "no preemptions to price"
    for req in preempted:
        assert req.prefill_tokens > req.prompt_len, (
            "resumed request did not recompute any tokens"
        )


def test_prompt_larger_than_pool_is_rejected_not_retried(model, encode):
    """A prompt that cannot fit in an empty pool must be rejected immediately.

    Waiting cannot help, so retrying it forever would hang the scheduler. This
    test exists because it did exactly that on the first attempt.
    """
    pool = PagedKVCache.for_model(model, num_blocks=3, block_size=4)  # 12 slots
    engine, results = run_engine(
        model, encode, [PROMPTS[2]], 60, pool=pool, max_batch_size=1
    )
    req = next(iter(results.values()))
    assert req.finish_reason == "aborted_prompt_too_long"
    assert req.output_token_ids == []
    assert engine.stats()["steps"] < 10, "scheduler spun instead of rejecting"
    assert engine.pool.manager.num_allocated == 0


def test_generation_outgrowing_pool_aborts_rather_than_truncating(model, encode):
    """A request whose prompt fits but whose output cannot must be reported.

    Silently returning a short answer would look like a normal completion.
    """
    pool = PagedKVCache.for_model(model, num_blocks=4, block_size=4)  # 16 slots
    engine, results = run_engine(
        model, encode, ["The capital of France is"], 60, pool=pool, max_batch_size=1
    )
    req = next(iter(results.values()))
    assert req.finish_reason == "aborted_cache_exhausted"
    assert 0 < len(req.output_token_ids) < 60
    assert engine.pool.manager.num_allocated == 0


# --- the batched decode step, in isolation ---

def test_decode_batch_matches_sequential_sequences(model, encode):
    """Two sequences of different length, batched vs stepped one at a time."""
    from kvengine import forward_with_own_cache

    pool = PagedKVCache.for_model(model, num_blocks=128, block_size=4)
    ids_a = encode(PROMPTS[0])
    ids_b = encode("Hi")

    seq_a, seq_b = pool.new_sequence("a"), pool.new_sequence("b")
    tok_a = int(torch.argmax(forward_with_own_cache(model, ids_a, seq_a)[0, -1, :]))
    tok_b = int(torch.argmax(forward_with_own_cache(model, ids_b, seq_b)[0, -1, :]))

    batch = plan_decode_batch([seq_a, seq_b], [tok_a, tok_b])
    assert batch.max_len == max(seq_a.length, seq_b.length) + 1
    assert batch.padding_waste() > 0, "expected padding with ragged lengths"
    batched_logits = forward_decode_batch(model, batch)

    # Same two steps, run independently in their own pool.
    solo_pool = PagedKVCache.for_model(model, num_blocks=128, block_size=4)
    solo_a, solo_b = solo_pool.new_sequence("a"), solo_pool.new_sequence("b")
    forward_with_own_cache(model, ids_a, solo_a)
    forward_with_own_cache(model, ids_b, solo_b)
    logit_a = forward_with_own_cache(
        model, torch.tensor([[tok_a]], device=ids_a.device), solo_a
    )[0, -1, :]
    logit_b = forward_with_own_cache(
        model, torch.tensor([[tok_b]], device=ids_b.device), solo_b
    )[0, -1, :]

    assert (batched_logits[0] - logit_a).abs().max().item() < 1e-3
    assert (batched_logits[1] - logit_b).abs().max().item() < 1e-3


def test_decode_batch_of_one_has_no_padding(model, encode):
    from kvengine import forward_with_own_cache

    pool = PagedKVCache.for_model(model, num_blocks=64, block_size=4)
    ids = encode(PROMPTS[0])
    seq = pool.new_sequence("solo")
    tok = int(torch.argmax(forward_with_own_cache(model, ids, seq)[0, -1, :]))
    batch = plan_decode_batch([seq], [tok])
    assert batch.padding_waste() == 0.0
    assert batch.batch_size == 1
