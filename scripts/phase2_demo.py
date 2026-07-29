"""Phase 2 demo: same tokens, same memory budget, far more concurrent requests.

Holds the memory budget fixed and asks how many requests each scheme can serve
with it. That is the comparison that matters on a real GPU, where the KV cache is
the binding constraint on throughput.

    python scripts/phase2_demo.py
    python scripts/phase2_demo.py --block-size 32 --reserve 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from kvengine import (  # noqa: E402
    DEFAULT_MODEL,
    ContiguousKVCache,
    PagedKVCache,
    greedy_own_cache,
    greedy_paged,
    hf_greedy,
    load_model,
)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "Q: What is 17 * 3?\nA:",
    "The three laws of thermodynamics are",
    "import numpy as np\n\ndef softmax(x):",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument(
        "--reserve",
        type=int,
        default=2048,
        help="per-request contiguous reservation the paged pool is compared against",
    )
    args = ap.parse_args()

    model, tokenizer = load_model(args.model, device=args.device, dtype=torch.float32)
    device = next(model.parameters()).device

    # Fix the memory budget at what the contiguous design needs for these prompts,
    # then give the paged pool exactly the same number of bytes.
    contiguous_probe = ContiguousKVCache.for_model(model, max_seq_len=args.reserve)
    per_request_bytes = contiguous_probe.nbytes()
    budget_bytes = per_request_bytes * len(PROMPTS)

    scratch = PagedKVCache.for_model(model, num_blocks=1, block_size=args.block_size)
    bytes_per_block = scratch.bytes_per_block()
    num_blocks = budget_bytes // bytes_per_block
    pool = PagedKVCache.for_model(model, num_blocks=num_blocks, block_size=args.block_size)

    print(f"model: {args.model}  device={device}  dtype=torch.float32")
    print(f"budget: {budget_bytes / 2**20:.1f} MiB "
          f"(= {len(PROMPTS)} contiguous requests at {args.reserve} slots)\n")
    print(f"contiguous: {per_request_bytes / 2**20:.1f} MiB per request, "
          f"{args.reserve} slots each")
    print(f"paged:      {num_blocks} blocks of {args.block_size} "
          f"({bytes_per_block / 2**20:.2f} MiB per block), "
          f"{num_blocks * args.block_size} slots total\n")

    all_match = True
    total_tokens = 0
    paged_slots_held = 0
    paged_secs = 0.0
    contig_secs = 0.0

    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        ref = hf_greedy(model, ids, max_new_tokens=args.max_new_tokens)
        contiguous = greedy_own_cache(
            model, ids, max_new_tokens=args.max_new_tokens, max_seq_len=args.reserve
        )
        paged = greedy_paged(
            model, ids, pool, max_new_tokens=args.max_new_tokens, free_on_finish=False
        )

        match = paged.new_token_ids == ref.new_token_ids == contiguous.new_token_ids
        all_match &= match
        used = len(ids[0]) + len(paged.new_token_ids)
        total_tokens += used
        paged_slots_held += paged.sequence.capacity
        paged_secs += paged.elapsed_s
        contig_secs += contiguous.elapsed_s

        print(f"prompt: {prompt!r}")
        print(f"  exact match (paged == contiguous == hf): {'YES' if match else 'NO'}")
        print(f"  {used} tokens | contiguous {contiguous.cache_utilization:6.1%} util "
              f"| paged {paged.cache_utilization:6.1%} util "
              f"({paged.blocks_held} blocks)")

    # Every request is still resident, as it would be mid-batch on a server.
    print()
    print("--- with all requests resident ---")
    contiguous_util = total_tokens / (len(PROMPTS) * args.reserve)
    paged_util = total_tokens / paged_slots_held
    print(f"real tokens held:        {total_tokens}")
    print(f"contiguous utilization:  {contiguous_util:6.1%}  "
          f"({len(PROMPTS) * args.reserve} slots reserved)")
    print(f"paged utilization:       {paged_util:6.1%}  "
          f"({paged_slots_held} slots held)")
    print(f"pool still free:         {pool.manager.num_free}/{pool.num_blocks} blocks")

    avg_tokens = total_tokens / len(PROMPTS)
    blocks_per_request = pool.manager.blocks_needed(int(avg_tokens))
    paged_capacity = pool.num_blocks // max(blocks_per_request, 1)
    print()
    print(f"--- concurrency at {budget_bytes / 2**20:.1f} MiB, "
          f"avg request {avg_tokens:.0f} tokens ---")
    print(f"contiguous: {len(PROMPTS)} concurrent requests")
    print(f"paged:      {paged_capacity} concurrent requests "
          f"({paged_capacity / len(PROMPTS):.0f}x more)")

    print()
    print(f"decode speed: contiguous {contig_secs:.2f}s total, paged {paged_secs:.2f}s "
          f"({paged_secs / contig_secs:.2f}x) -- gather costs memory traffic")

    pool.manager.check_invariants()
    print()
    print("ALL PROMPTS MATCH" if all_match else "MISMATCH FOUND")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
