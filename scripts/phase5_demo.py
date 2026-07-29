"""Phase 5 demo: speculative decoding speedup and acceptance rate.

Sweeps k, verifies every run is token-identical to plain greedy on the target, and
reports the acceptance rate and tokens-per-target-forward that explain the speedup.

    # self-draft sanity check (no extra download): acceptance should be ~100%
    python scripts/phase5_demo.py --target Qwen/Qwen2.5-0.5B --draft Qwen/Qwen2.5-0.5B

    # the real thing: 0.5B drafting for a 3B target
    python scripts/phase5_demo.py --target Qwen/Qwen2.5-3B --draft Qwen/Qwen2.5-0.5B \
        --out results/spec_decoding_mps.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import transformers  # noqa: E402

from kvengine import (  # noqa: E402
    PagedKVCache,
    greedy_paged,
    load_model,
    speculative_greedy,
)

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "The three laws of thermodynamics are",
    "import numpy as np\n\ndef softmax(x):",
    "Explain in one paragraph why the sky appears blue:",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    print(f"loading target {args.target} ...", flush=True)
    target, tokenizer = load_model(args.target, device=args.device, dtype=dtype)
    print(f"loading draft  {args.draft} ...", flush=True)
    draft, _ = load_model(args.draft, device=args.device, dtype=dtype)
    device = next(target.parameters()).device

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    print()
    print(f"device {device} | dtype {dtype}")
    print(f"target {args.target}: {n_params(target) / 1e9:.2f}B params, "
          f"{target.config.num_hidden_layers} layers")
    print(f"draft  {args.draft}: {n_params(draft) / 1e9:.2f}B params, "
          f"{draft.config.num_hidden_layers} layers")
    print(f"ratio  {n_params(target) / n_params(draft):.1f}x")
    print()

    # Separate pools: the two models have different layer counts and head dims.
    target_pool = PagedKVCache.for_model(target, args.blocks, args.block_size)
    draft_pool = PagedKVCache.for_model(draft, args.blocks, args.block_size)
    print(f"target pool {target_pool.nbytes() / 2**20:.1f} MiB | "
          f"draft pool {draft_pool.nbytes() / 2**20:.1f} MiB")
    print()

    encoded = [tokenizer(p, return_tensors="pt").input_ids.to(device) for p in PROMPTS]

    # --- baseline: plain greedy on the target ---
    print("baseline: plain greedy on target ...", flush=True)
    baseline = []
    for ids in encoded:
        baseline.append(greedy_paged(target, ids, target_pool,
                                     max_new_tokens=args.max_new_tokens))
    base_tokens = sum(len(r.new_token_ids) for r in baseline)
    base_time = sum(r.elapsed_s for r in baseline)
    base_tps = base_tokens / base_time
    print(f"  {base_tokens} tokens in {base_time:.2f}s = {base_tps:.1f} tok/s")
    print()

    rows = []
    for k in args.k:
        print(f"speculative, k={k} ...", flush=True)
        runs = []
        mismatches = []
        for i, (ids, ref) in enumerate(zip(encoded, baseline)):
            t_seq = target_pool.new_sequence("spec-target")
            d_seq = draft_pool.new_sequence("spec-draft")
            try:
                spec = speculative_greedy(
                    target, draft, ids, t_seq, d_seq,
                    k=k, max_new_tokens=args.max_new_tokens,
                )
            finally:
                t_seq.free()
                d_seq.free()
            runs.append(spec)
            if spec.new_token_ids != ref.new_token_ids:
                mismatches.append(PROMPTS[i])

        tokens = sum(len(r.new_token_ids) for r in runs)
        elapsed = sum(r.elapsed_s for r in runs)
        drafted = sum(r.tokens_drafted for r in runs)
        accepted = sum(r.tokens_accepted for r in runs)
        t_forwards = sum(r.target_forwards for r in runs)
        per_iter = [n for r in runs for n in r.accepted_per_iter]

        row = {
            "k": k,
            "tokens": tokens,
            "elapsed_s": round(elapsed, 3),
            "tokens_per_s": round(tokens / elapsed, 2),
            "speedup": round((tokens / elapsed) / base_tps, 3),
            "acceptance_rate": round(accepted / drafted, 4) if drafted else 0.0,
            "tokens_per_target_forward": round(tokens / t_forwards, 3),
            "target_forwards": t_forwards,
            "mean_accepted_per_iter": round(statistics.fmean(per_iter), 3) if per_iter else 0,
            "full_accept_iters": round(
                sum(1 for n in per_iter if n == k) / len(per_iter), 3
            ) if per_iter else 0,
            "exact_match": not mismatches,
        }
        rows.append(row)
        print(f"  {tokens} tokens in {elapsed:.2f}s = {row['tokens_per_s']:.1f} tok/s "
              f"({row['speedup']:.2f}x)  accept {row['acceptance_rate']:.1%}  "
              f"{row['tokens_per_target_forward']:.2f} tok/target-pass  "
              f"{'MATCH' if row['exact_match'] else 'MISMATCH'}")

    print()
    print("=" * 86)
    print(f"{'k':>3} {'tok/s':>8} {'speedup':>8} {'accept':>8} {'tok/pass':>9} "
          f"{'mean acc':>9} {'all-k iters':>12} {'exact':>7}")
    print(f"{'--':>3} {base_tps:8.1f} {'1.00x':>8} {'-':>8} {'1.00':>9} "
          f"{'-':>9} {'-':>12} {'ref':>7}")
    for r in rows:
        print(f"{r['k']:>3} {r['tokens_per_s']:8.1f} {r['speedup']:7.2f}x "
              f"{r['acceptance_rate']:7.1%} {r['tokens_per_target_forward']:9.2f} "
              f"{r['mean_accepted_per_iter']:9.2f} {r['full_accept_iters']:11.1%} "
              f"{'YES' if r['exact_match'] else 'NO':>7}")

    all_match = all(r["exact_match"] for r in rows)
    print()
    print("ALL RUNS IDENTICAL TO PLAIN GREEDY" if all_match else "MISMATCH FOUND")

    if args.out:
        payload = {
            "environment": {
                "target": args.target,
                "draft": args.draft,
                "target_params": n_params(target),
                "draft_params": n_params(draft),
                "device": str(device),
                "dtype": str(dtype),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "config": vars(args),
            "baseline": {
                "tokens": base_tokens,
                "elapsed_s": round(base_time, 3),
                "tokens_per_s": round(base_tps, 2),
            },
            "sweep": rows,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")

    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
