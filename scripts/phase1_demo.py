"""Phase 1 demo: our own cache matches HF, and wastes memory in a measurable way.

The reservation problem this exposes is the entire motivation for phase 2.

    python scripts/phase1_demo.py
    python scripts/phase1_demo.py --device mps --reserve 2048
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
    greedy_own_cache,
    greedy_with_cache,
    hf_greedy,
    load_model,
)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "Q: What is 17 * 3?\nA:",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument(
        "--reserve",
        type=int,
        default=2048,
        help="per-request contiguous reservation, as a real server would size it",
    )
    args = ap.parse_args()

    model, tokenizer = load_model(args.model, device=args.device, dtype=torch.float32)
    device = next(model.parameters()).device

    probe = ContiguousKVCache.for_model(model, max_seq_len=args.reserve)
    print(f"model: {args.model}  device={device}  dtype=torch.float32")
    print(f"cache: {tuple(probe.k.shape)} (layers, kv_heads, seq, head_dim)")
    print(f"       {probe.nbytes() / 2**20:.1f} MiB reserved per request "
          f"at {args.reserve} slots\n")

    all_match = True
    total_used = 0
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        ref = hf_greedy(model, ids, max_new_tokens=args.max_new_tokens)
        hf_cache = greedy_with_cache(model, ids, max_new_tokens=args.max_new_tokens)
        mine = greedy_own_cache(
            model, ids, max_new_tokens=args.max_new_tokens, max_seq_len=args.reserve
        )

        match = mine.new_token_ids == ref.new_token_ids == hf_cache.new_token_ids
        all_match &= match
        used = len(ids[0]) + len(mine.new_token_ids)
        total_used += used

        print(f"prompt: {prompt!r}")
        print(f"  -> {tokenizer.decode(mine.new_token_ids)!r}")
        print(f"  exact match (mine == hf == hf-cache): {'YES' if match else 'NO'}")
        print(f"  hf .generate()      {ref.elapsed_s:6.2f}s  {ref.tokens_per_s:6.1f} tok/s")
        print(f"  hf cache object     {hf_cache.elapsed_s:6.2f}s  {hf_cache.tokens_per_s:6.1f} tok/s")
        print(f"  my own cache        {mine.elapsed_s:6.2f}s  {mine.tokens_per_s:6.1f} tok/s")
        print(f"  cache slots used    {used}/{args.reserve} "
              f"= {mine.cache_utilization:.1%} utilization")
        print()

    n = len(PROMPTS)
    print(f"serving all {n} prompts concurrently with this design would reserve "
          f"{n * probe.nbytes() / 2**20:.1f} MiB")
    print(f"to hold {total_used} tokens of real data "
          f"({total_used / (n * args.reserve):.1%} utilization) "
          f"-- this is what phase 2 fixes")
    print()
    print("ALL PROMPTS MATCH" if all_match else "MISMATCH FOUND")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
