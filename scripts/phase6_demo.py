"""Phase 6 demo: prefix sharing vs recomputing every prompt from scratch.

Two workloads, because prefix caching pays off in two different ways and they are
worth separating:

  concurrent  many requests share one long system prompt and arrive together.
              The first one computes the prefix, the rest adopt its blocks.

  sequential  the same conversation is served twice, one after the other, with
              nothing in flight in between. This only works because a released
              block holding a registered prefix becomes *reclaimable* rather
              than free, so it survives the request that produced it.

Both runs use the same pool, the same prompts, and greedy decoding, so the
output must come out byte-identical. The saving shows up in prefill tokens, not
in what the model says.

    python scripts/phase6_demo.py
    python scripts/phase6_demo.py --blocks 128 --out results/prefix_sharing.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import transformers  # noqa: E402

from kvengine import (  # noqa: E402
    DEFAULT_MODEL,
    Engine,
    PagedKVCache,
    load_model,
)

# Long enough to span several blocks at block_size=16. This is the realistic shape
# of prefix traffic: a fixed instruction preamble followed by a short user turn.
SYSTEM = (
    "You are a careful and concise assistant. Answer directly, explain your "
    "reasoning in one sentence, and never speculate beyond the evidence given. "
    "If you are unsure about something, say so plainly instead of guessing. "
    "Prefer concrete examples over abstract description, and keep every answer "
    "under four sentences unless the user asks for more detail. "
)

QUESTIONS = [
    "What is the capital of France?",
    "Name three prime numbers.",
    "Why is the sky blue?",
    "What does a compiler do?",
    "How does a hash table work?",
    "What is the boiling point of water?",
]


def run_engine(model, prompts, *, share, blocks, block_size, max_batch_size, budget):
    """Serve every prompt through one engine instance and return outputs + stats."""
    engine = Engine(
        model,
        PagedKVCache.for_model(model, num_blocks=blocks, block_size=block_size),
        max_batch_size=max_batch_size,
        share_prefixes=share,
    )
    for i, ids in enumerate(prompts):
        engine.add_request(ids, max_new_tokens=budget, request_id=f"r{i}")
    t0 = time.perf_counter()
    finished = engine.run()
    elapsed = time.perf_counter() - t0
    engine.pool.manager.check_invariants()

    by_id = {r.request_id: r for r in finished}
    outputs = [by_id[f"r{i}"].output_token_ids for i in range(len(prompts))]
    return outputs, engine.stats(), elapsed


def run_sequential(model, prompts, *, share, blocks, block_size, budget):
    """Serve prompts one at a time through a single shared pool.

    One Engine, one request in flight at a time: each add_request/run pair fully
    drains before the next begins. Any prefix reuse here comes from blocks that
    outlived the request that created them.
    """
    engine = Engine(
        model,
        PagedKVCache.for_model(model, num_blocks=blocks, block_size=block_size),
        max_batch_size=1,
        share_prefixes=share,
    )
    outputs = []
    t0 = time.perf_counter()
    for i, ids in enumerate(prompts):
        engine.add_request(ids, max_new_tokens=budget, request_id=f"s{i}")
        finished = engine.run()
        outputs.append(finished[-1].output_token_ids)
    elapsed = time.perf_counter() - t0
    engine.pool.manager.check_invariants()
    return outputs, engine.stats(), elapsed


def report(name, off_stats, off_s, on_stats, on_s, matched):
    off_pf = off_stats["prefill_tokens"]
    on_pf = on_stats["prefill_tokens"]
    saved = off_pf - on_pf

    saved_frac = saved / off_pf if off_pf else 0.0

    print(f"--- {name} ---")
    print(f"prefill tokens, sharing off: {off_pf:6d}")
    print(f"prefill tokens, sharing on:  {on_pf:6d}  "
          f"({saved} fewer, {saved_frac:.1%} of prefill work skipped)")
    print(f"prompt tokens served from cache: {on_stats['prefix_tokens_reused']}")
    print(f"block lookup hit rate:           {on_stats['prefix_block_hit_rate']:.1%}")
    print(f"block evictions:                 {on_stats['prefix_block_evictions']}")
    print(f"wall clock: {off_s:.2f}s off -> {on_s:.2f}s on  ({off_s / on_s:.2f}x)")
    print(f"tokens generated: {off_stats['tokens_generated']} off, "
          f"{on_stats['tokens_generated']} on")
    print("OUTPUT IDENTICAL" if matched else "*** MISMATCH ***")
    print()

    return {
        "prefill_tokens_off": off_pf,
        "prefill_tokens_on": on_pf,
        "prefill_tokens_saved": saved,
        "prefill_saved_frac": round(saved_frac, 4),
        "prefix_tokens_reused": on_stats["prefix_tokens_reused"],
        "prefix_block_hit_rate": on_stats["prefix_block_hit_rate"],
        "prefix_block_evictions": on_stats["prefix_block_evictions"],
        "elapsed_s_off": round(off_s, 3),
        "elapsed_s_on": round(on_s, 3),
        "speedup": round(off_s / on_s, 3) if on_s else None,
        "tokens_generated": on_stats["tokens_generated"],
        "output_identical": matched,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tokenizer = load_model(args.model, device=args.device, dtype=torch.float32)
    device = next(model.parameters()).device

    prompts = [
        tokenizer(SYSTEM + q, return_tensors="pt").input_ids.to(device)[0].tolist()
        for q in QUESTIONS
    ]
    shared_len = len(tokenizer(SYSTEM, return_tensors="pt").input_ids[0])

    print(f"model: {args.model}  device={device}")
    print(f"pool: {args.blocks} blocks x {args.block_size}")
    print(f"workload: {len(prompts)} prompts, {shared_len}-token shared system prompt, "
          f"{sum(len(p) for p in prompts)} prompt tokens total\n")

    common = dict(blocks=args.blocks, block_size=args.block_size,
                  budget=args.max_new_tokens)

    # --- concurrent: all prompts in flight at once ---
    off_out, off_stats, off_s = run_engine(
        model, prompts, share=False, max_batch_size=args.max_batch_size, **common
    )
    on_out, on_stats, on_s = run_engine(
        model, prompts, share=True, max_batch_size=args.max_batch_size, **common
    )
    concurrent_match = off_out == on_out
    concurrent = report("concurrent", off_stats, off_s, on_stats, on_s, concurrent_match)

    # --- sequential: the same two conversations served back to back ---
    repeated = prompts + prompts
    soff_out, soff_stats, soff_s = run_sequential(
        model, repeated, share=False, **common
    )
    son_out, son_stats, son_s = run_sequential(model, repeated, share=True, **common)
    sequential_match = soff_out == son_out
    sequential = report(
        "sequential (each prompt served twice)",
        soff_stats, soff_s, son_stats, son_s, sequential_match,
    )

    all_match = concurrent_match and sequential_match

    if args.out:
        payload = {
            "environment": {
                "model": args.model,
                "device": str(device),
                "dtype": "float32",
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "config": vars(args),
            "workload": {
                "num_prompts": len(prompts),
                "shared_prefix_tokens": int(shared_len),
                "prompt_tokens_total": sum(len(p) for p in prompts),
            },
            "concurrent": concurrent,
            "sequential": sequential,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")

    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
