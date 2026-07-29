"""Phase 4: the benchmark harness itself.

A benchmark whose metrics are wrong is worse than no benchmark, because the
numbers get written down and believed. These tests cover the workload generator,
the percentile maths, and the baseline runner's accounting.
"""

import statistics

import pytest

from kvengine.bench import (
    RequestRecord,
    RunResult,
    _pct,
    make_workload,
    run_hf_baseline,
    run_kvengine,
    workload_summary,
)


# --- workload generation ---

def test_workload_is_deterministic_given_a_seed(tokenizer):
    a = make_workload(tokenizer, 12, seed=7)
    b = make_workload(tokenizer, 12, seed=7)
    assert [r.prompt_token_ids for r in a] == [r.prompt_token_ids for r in b]
    assert [r.max_new_tokens for r in a] == [r.max_new_tokens for r in b]


def test_different_seeds_give_different_workloads(tokenizer):
    a = make_workload(tokenizer, 12, seed=1)
    b = make_workload(tokenizer, 12, seed=2)
    assert [r.prompt_len for r in a] != [r.prompt_len for r in b]


def test_workload_respects_length_bounds(tokenizer):
    wl = make_workload(
        tokenizer, 40, seed=3, prompt_mean=100, output_mean=50,
        min_prompt=16, max_prompt=64, min_output=8, max_output=32,
    )
    for r in wl:
        assert 16 <= r.prompt_len <= 64, f"prompt out of bounds: {r.prompt_len}"
        assert 8 <= r.max_new_tokens <= 32
        assert len(r.prompt_token_ids) == r.prompt_len


def test_workload_lengths_are_right_skewed(tokenizer):
    """Lognormal, not uniform: mean above median, with a long tail.

    The spread is what drives padding waste and preemption, so a workload without
    it would flatter the engine.
    """
    wl = make_workload(tokenizer, 200, seed=5, prompt_mean=128, max_prompt=2048)
    lens = [r.prompt_len for r in wl]
    assert statistics.fmean(lens) > statistics.median(lens), "distribution not skewed"
    assert max(lens) > 2 * statistics.median(lens), "no long tail"


def test_workload_summary_reports_totals(tokenizer):
    wl = make_workload(tokenizer, 10, seed=0)
    s = workload_summary(wl)
    assert s["n_requests"] == 10
    assert s["prompt_tokens_total"] == sum(r.prompt_len for r in wl)
    assert s["output_tokens_requested"] == sum(r.max_new_tokens for r in wl)
    assert s["prompt_len"]["min"] <= s["prompt_len"]["p50"] <= s["prompt_len"]["max"]


# --- metrics ---

def test_percentile_is_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _pct(values, 50) == 5
    assert _pct(values, 90) == 9
    assert _pct(values, 100) == 10
    assert _pct(values, 1) == 1


def test_percentile_handles_single_value_and_empty():
    assert _pct([42], 50) == 42
    assert _pct([42], 99) == 42
    assert _pct([], 50) == 0.0


def test_summary_computes_throughput_from_output_tokens():
    records = [
        RequestRecord("a", prompt_len=10, output_len=20, ttft_s=0.1,
                      latency_s=1.0, tpot_s=0.05, finish_reason="length"),
        RequestRecord("b", prompt_len=30, output_len=40, ttft_s=0.3,
                      latency_s=2.0, tpot_s=0.05, finish_reason="eos"),
    ]
    s = RunResult("test", wall_s=2.0, records=records).summary()
    assert s["output_tokens"] == 60
    assert s["prompt_tokens"] == 40
    assert s["output_tokens_per_s"] == 30.0
    assert s["total_tokens_per_s"] == 50.0
    assert s["requests_per_s"] == 1.0
    assert s["latency_s"]["p50"] == 1.0
    assert s["latency_s"]["p99"] == 2.0


def test_summary_tolerates_missing_latencies():
    """An aborted request has no finish time, and must not crash the summary."""
    records = [
        RequestRecord("a", 10, 0, None, None, None, "aborted_prompt_too_long"),
        RequestRecord("b", 10, 5, 0.2, 1.0, 0.1, "length"),
    ]
    s = RunResult("test", wall_s=1.0, records=records).summary()
    assert s["requests"] == 2
    assert s["latency_s"]["p50"] == 1.0


# --- end to end, small scale ---

def test_run_kvengine_records_every_request(model, tokenizer):
    wl = make_workload(tokenizer, 4, seed=11, prompt_mean=24, output_mean=8,
                       max_prompt=32, max_output=10)
    result = run_kvengine(model, wl, num_blocks=128, block_size=16, max_batch_size=4)

    assert len(result.records) == len(wl)
    assert {r.request_id for r in result.records} == {r.request_id for r in wl}
    for rec in result.records:
        assert rec.ttft_s is not None and rec.ttft_s >= 0
        assert rec.latency_s >= rec.ttft_s, "finished before first token"
    s = result.summary()
    assert s["output_tokens"] > 0
    assert "scheduler" in s and "cache" in s


def test_run_kvengine_respects_per_request_budgets(model, tokenizer):
    wl = make_workload(tokenizer, 4, seed=12, prompt_mean=24, output_mean=8,
                       max_prompt=32, max_output=12)
    result = run_kvengine(model, wl, num_blocks=128, max_batch_size=4)
    budgets = {r.request_id: r.max_new_tokens for r in wl}
    for rec in result.records:
        assert rec.output_len <= budgets[rec.request_id], "exceeded its token budget"


def test_baseline_respects_per_request_budgets(model, tokenizer):
    """Static batching generates to the batch max, then must truncate per request."""
    wl = make_workload(tokenizer, 4, seed=13, prompt_mean=24, output_mean=8,
                       max_prompt=32, max_output=12)
    result = run_hf_baseline(model, wl, batch_size=4)
    budgets = {r.request_id: r.max_new_tokens for r in wl}
    assert len(result.records) == len(wl)
    for rec in result.records:
        assert rec.output_len <= budgets[rec.request_id], "exceeded its token budget"
        assert rec.finish_reason in {"length", "eos"}


def test_baseline_unbatched_matches_engine_exactly(model, tokenizer):
    """With batch_size=1 there is no padding, so both paths must agree token-wise.

    This is what makes the throughput comparison meaningful: the two systems are
    doing the same work, not merely similar work.
    """
    wl = make_workload(tokenizer, 3, seed=14, prompt_mean=24, output_mean=10,
                       max_prompt=32, max_output=12)
    engine = run_kvengine(model, wl, num_blocks=128, max_batch_size=1)
    baseline = run_hf_baseline(model, wl, batch_size=1, sort_by_length=False)

    e = {r.request_id: r.output_len for r in engine.records}
    b = {r.request_id: r.output_len for r in baseline.records}
    assert e == b, "engine and unbatched baseline generated different lengths"


def test_baseline_latency_is_measured_from_arrival_not_batch_start(model, tokenizer):
    """Both systems must time from t0, or the latency comparison is meaningless.

    With more requests than fit in one batch, a request in the second batch waited
    through the first. Timing it from its own batch's start would hide that queue
    delay while the engine honestly reports its own.
    """
    wl = make_workload(tokenizer, 6, seed=17, prompt_mean=24, output_mean=8,
                       max_prompt=32, max_output=10)
    result = run_hf_baseline(model, wl, batch_size=2)  # 3 batches

    latencies = sorted(r.latency_s for r in result.records)
    assert latencies[-1] > latencies[0], (
        "all requests reported the same latency, so queue delay was not counted"
    )
    assert latencies[-1] <= result.wall_s + 1e-6, "latency exceeded total wall time"


def test_baseline_reports_cache_utilization_below_engine(model, tokenizer):
    """The core memory claim, on a workload with real length spread."""
    wl = make_workload(tokenizer, 8, seed=15, prompt_mean=32, output_mean=24,
                       max_prompt=96, max_output=48)
    engine = run_kvengine(model, wl, num_blocks=256, max_batch_size=8)
    baseline = run_hf_baseline(model, wl, batch_size=8)

    e_util = engine.summary()["cache"]["slot_utilization"]
    b_util = baseline.summary()["cache"]["slot_utilization"]
    assert 0 < b_util < e_util <= 1.0, (
        f"expected paged utilization to beat static batching: {b_util} vs {e_util}"
    )


def test_run_result_json_is_serializable(model, tokenizer):
    import json

    wl = make_workload(tokenizer, 2, seed=16, prompt_mean=16, output_mean=6,
                       max_prompt=24, max_output=8)
    result = run_kvengine(model, wl, num_blocks=64, max_batch_size=2)
    text = json.dumps(result.to_json())
    assert '"summary"' in text and '"requests"' in text
