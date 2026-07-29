"""Phase 3 demo: continuous batching vs serving one request at a time.

Same pool, same prompts, same output. The only difference is whether requests
share a forward pass.

    python scripts/phase3_demo.py
    python scripts/phase3_demo.py --max-batch-size 4 --blocks 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from kvengine import (  # noqa: E402
    DEFAULT_MODEL,
    Engine,
    PagedKVCache,
    greedy_paged,
    load_model,
)

# Deliberately uneven prompt lengths and generation budgets, so requests retire at
# different times and the batch has to refill mid-run.
WORKLOAD = [
    ("The capital of France is", 40),
    ("def fibonacci(n):", 56),
    ("Once upon a time, in a small village by the sea,", 48),
    ("Q: What is 17 * 3?\nA:", 24),
    ("The three laws of thermodynamics are", 40),
    ("import numpy as np\n\ndef softmax(x):", 56),
    ("Hi", 16),
    ("Explain why the sky is blue in one sentence:", 32),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch-size", type=int, default=8)
    args = ap.parse_args()

    model, tokenizer = load_model(args.model, device=args.device, dtype=torch.float32)
    device = next(model.parameters()).device
    encoded = [
        (tokenizer(p, return_tensors="pt").input_ids.to(device), n) for p, n in WORKLOAD
    ]

    print(f"model: {args.model}  device={device}")
    print(f"pool: {args.blocks} blocks x {args.block_size}  "
          f"max_batch_size={args.max_batch_size}")
    print(f"workload: {len(WORKLOAD)} requests, "
          f"{sum(n for _, n in WORKLOAD)} tokens requested\n")

    # --- baseline: one request at a time through the same paged pool ---
    pool = PagedKVCache.for_model(model, num_blocks=args.blocks, block_size=args.block_size)
    seq_outputs = []
    t0 = time.perf_counter()
    for ids, budget in encoded:
        seq_outputs.append(greedy_paged(model, ids, pool, max_new_tokens=budget))
    sequential_s = time.perf_counter() - t0
    sequential_tokens = sum(len(r.new_token_ids) for r in seq_outputs)

    # --- continuous batching ---
    engine = Engine(
        model,
        PagedKVCache.for_model(model, num_blocks=args.blocks, block_size=args.block_size),
        max_batch_size=args.max_batch_size,
    )
    for i, (ids, budget) in enumerate(encoded):
        engine.add_request(ids[0].tolist(), max_new_tokens=budget, request_id=f"r{i}")
    t0 = time.perf_counter()
    finished = engine.run()
    batched_s = time.perf_counter() - t0
    stats = engine.stats()

    by_id = {r.request_id: r for r in finished}
    matches = all(
        by_id[f"r{i}"].output_token_ids == seq_outputs[i].new_token_ids
        for i in range(len(encoded))
    )

    print("--- throughput ---")
    print(f"one at a time:      {sequential_s:6.2f}s  "
          f"{sequential_tokens / sequential_s:6.1f} tok/s")
    print(f"continuous batching {batched_s:6.2f}s  "
          f"{stats['tokens_generated'] / batched_s:6.1f} tok/s  "
          f"({sequential_s / batched_s:.2f}x)")
    print()
    print("--- scheduler ---")
    print(f"steps:                {stats['steps']} "
          f"({stats['prefill_steps']} prefill, {stats['decode_steps']} decode, "
          f"{stats['idle_steps']} idle)")
    print(f"avg decode batch:     {stats['avg_decode_batch']:.2f} "
          f"(max {stats['max_decode_batch']})")
    print(f"padding waste:        {stats['avg_padding_waste']:.1%} "
          f"of gathered K/V (ragged lengths)")
    print(f"peak pool used:       {stats['peak_pool_utilization']:.1%} of blocks")
    print(f"slot utilization:     {stats['avg_slot_utilization']:.1%} "
          f"of held slots hold real tokens")
    print(f"preemptions:          {stats['preemptions']}")
    print(f"tokens generated:     {stats['tokens_generated']}")

    print()
    print("--- per request (arrival -> first token -> finish, in steps) ---")
    for i in range(len(encoded)):
        r = by_id[f"r{i}"]
        prompt = WORKLOAD[i][0].replace("\n", "\\n")[:34]
        print(f"  {r.request_id:4s} {prompt:36s} "
              f"ttft={r.first_token_step - r.arrival_step:3d} "
              f"finish={r.finish_step:4d} "
              f"tokens={len(r.output_token_ids):3d} "
              f"{r.finish_reason}")

    engine.pool.manager.check_invariants()
    print()
    print("OUTPUT IDENTICAL TO ONE-AT-A-TIME" if matches else "MISMATCH FOUND")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
