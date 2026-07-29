# kvengine

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
| 4 | Benchmark vs tuned HF baseline on a T4 | next |
| 5 | Speculative decoding | |

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 72 tests
.venv/bin/python scripts/phase2_demo.py       # memory: paging vs contiguous
.venv/bin/python scripts/phase3_demo.py       # throughput: batched vs one-at-a-time
```

## The result so far

Same memory budget, same model, same output. The only difference is how the KV
cache is laid out.

```
budget: 288 MiB   (= 6 contiguous requests reserving 2048 slots each)

                  utilization    concurrent requests
contiguous              2.4%                      6
paged                  87.8%                    192   (32x)
```

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

## Gotchas hit along the way

**`.generate(do_sample=False)` is not pure argmax.** Qwen ships a
`generation_config.json` carrying sampling defaults, including
`repetition_penalty`. Logits warpers are applied regardless of `do_sample`, so
comparing a clean argmax loop against a naive `.generate()` call shows
mismatches that look exactly like a cache bug. `reference.py` builds an explicit
`GenerationConfig` with every processor disabled, so "my loop matches
`.generate()`" is a claim about the cache and nothing else.

**RoPE bakes position into the key permanently.** Position isn't stored
alongside the key, it's *rotated into* it at write time. A key written at the
wrong position stays silently wrong forever and poisons every later token that
attends to it. This is why offsets are tracked so pedantically.

**The causal mask offset during prefill.** New queries sit at the *end* of the
timeline: query `i` is at absolute position `start + i`, not `i`. A single decode
token needs no mask at all. There's a test asserting the exact triangular pattern
for 3 queries on top of 5 cached tokens.

**Blocks are never zeroed on free.** If a stale read existed, a zeroed buffer
would hide it. Two tests exploit this: reusing a cache without zeroing must match
a fresh run, which proves nothing past `length` is ever read.

**Contiguous block tables hide gather bugs.** Blocks handed out in ascending
order make a broken gather look correct. One test fragments the pool first and
asserts the block table is out of order *before* checking output; another forces a
descending table to catch a gather that assumes ascending physical ids.

**The resume boundary after preemption is off-by-one bait.** A resumed request
must prefill prompt + generated[:-1], not prompt + generated: the last generated
token has not yet been fed to the model, so including it would consume it twice
and shift the whole output by one. The cache invariant is
`length == prompt_len + len(generated) - 1`.

**A prompt too large for the pool hung the scheduler.** Admission checked
`needed > num_free` and broke out of the loop to wait — but for a prompt bigger
than the entire pool, waiting can never help, so the loop spun forever with
nothing running. Admission now rejects anything that would not fit in an *empty*
pool, and `step()` carries a backstop that aborts the queue head when a step is
idle with nothing running. Both are tested.

**`padding_waste` depended on when it was called.** It derived history lengths
from the live sequences, but `forward_decode_batch` commits, so the answer changed
before vs after the forward pass. Lengths are now snapshotted at plan time. A
metric that silently means two different things is worse than no metric.

**HuggingFace Xet downloads hang on this machine.** Metadata comes from
`huggingface.co` and works fine; only the weight file goes through Xet, so it
presents as a silent 0-byte stall rather than an error. `loader.py` sets
`HF_HUB_DISABLE_XET=1` to force the ordinary HTTP path.

## Known differences from vLLM

**Gather materializes.** `write()` returns the full K/V history for a layer as a
contiguous tensor, then runs standard attention over it. vLLM instead ships a
fused CUDA kernel that attends block-by-block and never materializes the history,
so it pays no extra memory traffic. This implementation is the honest portable
version: correct, and slower by a constant. That constant is the 1.07x above.

**No prefix sharing.** Two requests with a common prefix each get their own
blocks. vLLM supports copy-on-write block sharing for exactly this; the block
allocator here already tracks ownership, which is the hook that would need to
become a reference count.

**Decode batches are padded, not ragged.** vLLM's varlen kernels consume a ragged
batch directly. Padding to the longest history wastes work proportional to the
spread of lengths — measured at 15.3% on the demo workload.

**No chunked prefill.** Prefill and decode occupy separate steps here, so a long
prompt stalls decoding for a step. See the scheduling discussion above.

**vLLM vs SGLang.** Both attack KV memory, differently. vLLM's paged attention
solves *internal fragmentation* — dynamic sizing via a block table, waste bounded
by one block per request. SGLang's RadixAttention solves *redundant computation* —
a radix tree over token prefixes lets requests sharing a prefix share the cached
KV for it, with LRU eviction. They compose: paging is a memory allocator, radix
prefix caching is a cache with a hit rate. Paging always helps; radix caching
helps in proportion to how much prefix traffic actually repeats (system prompts,
few-shot examples, multi-turn chat).

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
tests/
  test_phase0.py  exact match vs .generate()
  test_phase1.py  own cache, causal mask, chunked prefill
  test_blocks.py  allocator churn, OOM, double-ownership
  test_phase2.py  paged correctness, fragmentation, interleaved requests
  test_phase3.py  batching invariance, ragged masks, preemption, rejection
scripts/
  phase0_demo.py .. phase3_demo.py
```
