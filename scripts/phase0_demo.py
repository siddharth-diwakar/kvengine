"""Phase 0 demo: show the three decode paths agree, and what caching buys.

    python scripts/phase0_demo.py
    python scripts/phase0_demo.py --device mps --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from kvengine import (  # noqa: E402
    DEFAULT_MODEL,
    greedy_no_cache,
    greedy_with_cache,
    hf_greedy,
    load_model,
    model_shape_info,
)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    model, tokenizer = load_model(args.model, device=args.device, dtype=torch.float32)
    info = model_shape_info(model)
    print(f"model: {args.model}")
    print(f"  device={info['device']} dtype={info['dtype']}")
    print(f"  layers={info['num_layers']} q_heads={info['num_attention_heads']} "
          f"kv_heads={info['num_key_value_heads']} head_dim={info['head_dim']}")
    print()

    all_match = True
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(info["device"])

        ref = hf_greedy(model, ids, max_new_tokens=args.max_new_tokens)
        cached = greedy_with_cache(model, ids, max_new_tokens=args.max_new_tokens)
        uncached = greedy_no_cache(model, ids, max_new_tokens=args.max_new_tokens)

        match = cached.new_token_ids == ref.new_token_ids == uncached.new_token_ids
        all_match &= match

        print(f"prompt: {prompt!r}")
        print(f"  -> {tokenizer.decode(cached.new_token_ids)!r}")
        print(f"  exact match (mine == hf == no-cache): {'YES' if match else 'NO'}")
        print(f"  hf .generate()      {ref.elapsed_s:6.2f}s  {ref.tokens_per_s:6.1f} tok/s")
        print(f"  mine, with cache    {cached.elapsed_s:6.2f}s  {cached.tokens_per_s:6.1f} tok/s")
        print(f"  mine, no cache      {uncached.elapsed_s:6.2f}s  {uncached.tokens_per_s:6.1f} tok/s"
              f"   ({uncached.elapsed_s / cached.elapsed_s:.1f}x slower)")
        print()

    print("ALL PROMPTS MATCH" if all_match else "MISMATCH FOUND")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
