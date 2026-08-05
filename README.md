# kvengine

[![ci](https://github.com/siddharth-diwakar/kvengine/actions/workflows/ci.yml/badge.svg)](https://github.com/siddharth-diwakar/kvengine/actions/workflows/ci.yml)

A small LLM inference engine built from scratch, to understand how production
serving engines actually work. Target model is Qwen2.5-0.5B; everything runs on a
Mac (CPU or MPS) and is meant to port to a T4 for benchmarking.

The engine is built in layers, and **every layer is checked by the same test:
greedy decoding must produce token-for-token identical output to HuggingFace's
`.generate()`.** That single anchor is what makes it possible to replace the KV
cache with progressively more aggressive machinery without wondering whether the
model still works.

| Phase | What it owns | Status |
|---|---|---|
| 0 | Manual decode loop, HuggingFace's cache object | done |
| 1 | Own KV tensor, own attention math | done |
| 2 | Paged block allocator + gather from scattered blocks | done |
| 3 | Continuous batching: scheduler, preemption, batched decode | done |
| 4 | Scripted benchmark vs tuned HF baseline, results in JSON | done (T4 run pending) |
| 5 | Speculative decoding: 0.5B draft for a 3B target | done |
| 6 | Prefix sharing: refcounted blocks + content-addressed cache | done |

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -m fast            # 35 tests, no weights, 0.1s
.venv/bin/python -m pytest                    # 127 tests, ~3 min
.venv/bin/python scripts/phase2_demo.py       # memory: paging vs contiguous
.venv/bin/python scripts/phase3_demo.py       # throughput: batched vs one-at-a-time
.venv/bin/python scripts/benchmark.py --requests 24 --out results/local.json
.venv/bin/python scripts/phase5_demo.py --dtype float16   # spec decoding k-sweep
.venv/bin/python scripts/phase6_demo.py       # prefix sharing: prefill saved
```

The Phase 5 sweep downloads Qwen2.5-3B (~6 GB). Pass
`--target Qwen/Qwen2.5-0.5B --draft Qwen/Qwen2.5-0.5B` to exercise the code path
with no extra download; acceptance will be 100% and it will be slower, for the
reason in the gotchas.

## The results so far

**Memory.** Same budget, same model, same output; only the cache layout differs.

```
budget: 288 MiB   (= 6 contiguous requests reserving 2048 slots each)

                  utilization    concurrent requests
contiguous              2.4%                      6
paged                  87.8%                    192   (32x)
```

**End to end vs a tuned HuggingFace baseline.** 24 requests fired at once,
lognormal lengths, Qwen2.5-0.5B fp32 on an M4 Pro (MPS). Baseline is static
batching, length-sorted, batch 16 — see [the benchmark section](#benchmark-benchpy-scriptsbenchmarkpy)
for why that is a fair fight.

```
                        huggingface     kvengine
wall                         17.51s       15.53s
output tok/s                  170.4        192.1     1.13x
cache utilization             56.9%        94.3%     1.66x
p90 latency                  17.51s       13.32s
p90 time-to-first-token      11.90s        5.88s
p99 inter-token latency       0.347s       0.075s    4.6x
```

On a prompt-heavy workload (mean prompt 128, mean output 64) throughput is a wash
and the latency picture is the same shape:

```
                        huggingface     kvengine
output tok/s                  119.9        120.9     1.01x
cache utilization             58.7%        96.6%     1.65x
p90 latency                  16.83s       14.95s
p90 time-to-first-token      10.79s        7.77s
p99 inter-token latency       0.484s       0.158s    3.1x
```

Throughput parity here is expected and is the clearest remaining weakness: my
prefill is per-request while HuggingFace prefills a whole batch of prompts in one
pass, so the more prompt-heavy the workload, the more that costs.

The throughput gain is modest (1.13x, or nil when prompt-heavy) but the **tail
latency difference is large**, and that is the more honest headline. Static batching makes a request
wait for its whole batch: the baseline's p90 TTFT is 11.9s because requests in the
second batch sit idle until the first batch's longest member finishes. Continuous
batching admits them as slots free up.

The p99 inter-token latency gap (4.6x) is the same effect from the user's side: in
a static batch, as sequences finish, the batch keeps running at full width with
mostly-dead rows, so per-token time degrades. Here, finished requests leave.

**These are CPU/MPS numbers on a 0.5B model, and they understate the throughput
case.** Batched decode wins by amortising the weight read across requests, which
pays off when decode is memory-bandwidth-bound — true on a GPU, much less true for
fp32 matmuls on Apple silicon. The memory win (1.66x utilization here, 32x
concurrency in the phase 2 microbenchmark) is what converts to throughput on real
hardware, because it is what lets you run the larger batch in the first place.

Raw results with full per-request records are in `results/`. The fp16 code path is
smoke-tested; T4 numbers pending an actual T4.

**Speculative decoding.** Qwen2.5-0.5B drafting for Qwen2.5-3B (6.2x parameter
ratio), fp16 on MPS, 6 prompts x 64 tokens. Every run is token-identical to plain
greedy on the 3B target.

```
  k    tok/s   speedup   accept   tok/target-pass   all-k iters
 --     21.8     1.00x        -              1.00             -
  1     25.8     1.19x    82.5%              1.81         82.5%
  2     32.9     1.51x    76.6%              2.49         69.5%
  4     35.2     1.61x    64.7%              3.52         53.2%
  8     29.4     1.35x    48.3%              4.68         26.8%
```

**k=4 is the optimum, and k=8 is worse than k=2.** The curve is non-monotonic
because k trades two things against each other: more drafted tokens means more
tokens per target pass (1.81 → 4.68, monotonically better) but a longer speculation
runs further from the target's own distribution, so acceptance falls (82.5% →
48.3%) while drafting cost grows linearly in k. At k=8 more than half of every
draft is thrown away, and the draft passes that produced it are pure waste. The
fraction of iterations where *all* k are accepted collapses from 82.5% to 26.8%,
which is the clearest single indicator of where the knee is.

**Prefix sharing.** 6 prompts behind a 63-token shared system prompt, Qwen2.5-0.5B
fp32 on CPU, greedy. Output is byte-identical with sharing on and off.

```
                        prefill tokens    saved    block hit rate   wall clock
concurrent (6 at once)      411 -> 155    62.3%             76.2%        1.12x
sequential (each twice)     822 -> 182    77.9%             88.9%        1.04x
```

**The prefill saving is large and the wall-clock saving is not, which is the
interesting part.** 62% of prefill work disappears, but total time barely moves,
because on a 0.5B model at fp32 on CPU these prompts are short and decode
dominates the run — cutting prefill cuts a small slice of the whole. Prefix
caching pays in proportion to how prompt-heavy the traffic is, and this workload
is not. The regimes where it actually wins are long system prompts, few-shot
examples, and multi-turn chat where each turn re-sends the entire conversation;
that is the same reason the number to watch is prefill tokens saved, not the
speedup on this particular workload.

The sequential row is the one that needed real machinery. Sharing between
*concurrent* requests only needs a refcount. Sharing across requests that never
overlap needs freed blocks to survive their owner, which is what the reclaimable
state below is for.

Utilization is the fraction of reserved cache memory actually holding real
tokens. The contiguous scheme must reserve for the worst case *before* it knows
how long the answer will be — one prompt here answered in 17 tokens and still
held a 2048-slot reservation, for **0.8% utilization**. Paging hands out 16-token
blocks on demand, so waste is capped at under one block per request regardless of
final length.

Cost of paging: **1.07x slower decode** in this implementation, because gathering
scattered blocks moves more memory. See [known differences from
vLLM](#known-differences-from-vllm) — a fused kernel removes most of that.

## Design

### Cache layout

```
[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
      24          768         16            2           64
```

`num_kv_heads`, not `num_attention_heads`. Qwen2.5 uses grouped-query attention:
14 query heads share 2 KV heads, so KV heads are duplicated 7x at attention time
(`repeat_kv`). Sizing the cache off query heads would cost 7x the memory and read
the wrong slots — it's the first mistake to make here, so `model_shape_info()`
surfaces both numbers and a test asserts they differ.

Blocks and slots-within-block are adjacent dimensions on purpose. Merging them
gives a flat slot index space where `slot = block_id * block_size + offset`, so
writes are a single `index_copy_` into a flat view. vLLM lays its cache out the
same way for the same reason.

### Block allocator (`blocks.py`)

A slab pool: fixed-size blocks, a LIFO free list, `allocate`/`free`, and an owner
map. Deliberately contains no tensors, so it can be tested exhaustively without
loading a model.

Two design choices worth defending:

- **`allocate()` is atomic.** Capacity is checked before a single block is popped.
  A partial allocation that then raises would leak blocks on every rejected
  request, and a loaded server rejects requests constantly — the pool would bleed
  capacity until it wedged.
- **LIFO, not FIFO.** The most recently freed block is most likely still warm in
  cache, and reusing it immediately makes use-after-free bugs fail loudly in tests
  instead of hiding behind stale-but-plausible data.

`check_invariants()` asserts no block is both free and allocated, no duplicates in
the free list, and that free + allocated covers the pool. A 4000-operation
randomized churn test calls it after *every single operation*, and separately
asserts no block ever appears in two live block tables.

### Paged attention (`paged.py`)

`PagedSequenceCache` exposes exactly the interface the Phase 1 contiguous cache
does — `length` / `reserve` / `write` / `commit`. So the forward pass from Phase 1
drives either cache **without a single change**: paging is entirely hidden behind
`write()`. That's the cleanest evidence the abstraction is right.

The ordering contract matters: `reserve()` once, then `write()` once per layer at
the same `start`, then `commit()` once. All 24 layers write at the same offset in
a forward pass, so growth cannot happen inside the layer loop and the length
cannot advance until every layer is done. Getting this wrong slides the offset out
from under layer 1.

### Continuous batching (`engine.py`, `batch.py`)

8 uneven requests, one pool, identical output:

```
one at a time:        6.17s   45.0 tok/s
continuous batching:  4.45s   62.5 tok/s   (1.39x)
avg decode batch 4.91 (max 8) | padding waste 15.3% | 0 preemptions
```

**The 1.39x is a CPU number and understates the point.** Batched decode wins by
amortising the model weight read across requests, which pays off when decode is
memory-bandwidth-bound — true on a GPU, much less true on a CPU where fp32
matmuls are compute-bound. Phase 4 measures this properly on a T4.

**The scheduling decision.** New requests need a prefill pass over the whole
prompt; running requests need a one-token decode pass. They cannot share a
forward pass without extra machinery, so the scheduler must pick an order:

- `prefill_first` (default) minimises time-to-first-token, but each prefill stalls
  every running request for a step, making inter-token latency spiky.
  `max_prefills_per_step` bounds the damage.
- `decode_first` gives smooth inter-token latency and the best decode throughput,
  at the cost of new arrivals waiting behind long generations.

Production engines dodge the dilemma with *chunked prefill*: split a long prompt
into fixed-size pieces and mix one piece into a decode batch so neither side
stalls. The correctness prerequisite already passes here
(`test_incremental_prefill_equals_single_prefill`); what is missing is a forward
pass accepting ragged prefill chunks and decode tokens together.

**Preemption.** A decode step can fail to allocate when every running request
wants one more slot and the pool is full. The scheduler frees the *most recently
admitted* request's blocks and returns it to the queue, so older requests keep
progressing (FCFS fairness). Its generated tokens are kept, and on re-admission
the prompt plus those tokens are prefilled again to rebuild the cache. This is
vLLM's recompute preemption — it costs a prefill but needs no extra memory, which
is the right trade when memory is exactly what ran out. (vLLM's alternative is
swapping blocks to host memory.) A test runs 4 requests through a 14-block pool
and asserts preemption actually fires *and* that output is still byte-identical.

Batched decode has one query token per request but a different history length per
request, so gathered K/V is padded to the longest history in the batch and the
padding masked out. Padding waste is reported (15.3% above) because it is the
number that justifies grouping requests of similar length.

### Benchmark (`bench.py`, `scripts/benchmark.py`)

Scripted, not notebook-interactive, and device-agnostic so the same command
produces the local numbers and the T4 numbers. Results are written to JSON under
`results/`, tagged with device, dtype, library versions and git commit.

```bash
# local
python scripts/benchmark.py --requests 24 --prompt-mean 32 --output-mean 96 \
    --blocks 512 --max-batch-size 16 --baseline-batch-size 16 \
    --out results/benchmark_mps_decode_heavy.json

# Kaggle T4 (fp16)
python scripts/benchmark.py --device cuda --dtype float16 \
    --requests 128 --prompt-mean 256 --output-mean 128 \
    --blocks 2048 --max-batch-size 32 --baseline-batch-size 32 \
    --out results/benchmark_t4.json
```

**What makes the baseline a fair fight.** Comparing against one-at-a-time
`.generate()` would be a strawman, so the baseline gets fp16, static batching, and
length-sorted batches to minimise padding. What it structurally cannot do is
refill a slot when a request finishes early — a static batch runs until its
longest member is done. That gap is what continuous batching closes, and it should
show up in the numbers rather than be asserted.

Workload lengths are lognormal, not uniform: real traffic is right-skewed, and the
*spread* is what drives padding waste and preemption. A uniform workload would
hide exactly the behaviour being measured. Fixed seed, so runs are comparable.
=



## Layout

```
kvengine/
  loader.py     model + tokenizer loading, cache-relevant shape info
  decode.py     four greedy loops: no-cache, HF cache, own cache, paged
  reference.py  strict-greedy HuggingFace baseline (the correctness anchor)
  forward.py    hand-written forward pass: RoPE, causal mask, GQA, attention
  cache.py      phase 1: one contiguous buffer per request
  blocks.py     phase 2: the block allocator (no tensors)
  paged.py      phase 2: block-backed cache + scatter/gather
  batch.py      phase 3: batched decode step (padded gather + mask)
  engine.py     phase 3: scheduler, admission, preemption, metrics
  bench.py      phase 4: workload generator, tuned HF baseline, metrics
  speculative.py phase 5: draft/verify loop with cache rollback
  blocks.py     phase 6 also: refcounts, reclaimable state, prefix registry
  paged.py      phase 6 also: adopt_prefix / register_prefix_blocks
results/        phases 4-6: committed benchmark JSON, one file per run
tests/
  test_phase0.py  exact match vs .generate()
  test_phase1.py  own cache, causal mask, chunked prefill
  test_blocks.py  allocator churn, OOM, double-ownership
  test_phase2.py  paged correctness, fragmentation, interleaved requests
  test_phase3.py  batching invariance, ragged masks, preemption, rejection
  test_phase4.py  workload determinism, percentile maths, baseline accounting
  test_phase5.py  spec decoding vs greedy under self/hostile drafts, truncate
  test_phase6.py  refcount churn, LRU eviction, sharing changes no output
scripts/
  phase0_demo.py .. phase3_demo.py, phase5_demo.py, phase6_demo.py, benchmark.py
```
