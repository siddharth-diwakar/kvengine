"""Phase 4: scripted benchmark. kvengine vs a tuned HuggingFace baseline.

Device-agnostic on purpose so the same script produces the local CPU/MPS numbers
and the Kaggle T4 numbers, and the JSON records which it was.

    # local
    python scripts/benchmark.py --requests 24 --out results/benchmark_cpu.json

    # Kaggle T4
    python scripts/benchmark.py --device cuda --dtype float16 \
        --requests 128 --prompt-mean 256 --output-mean 128 \
        --blocks 2048 --max-batch-size 32 --baseline-batch-size 32 \
        --out results/benchmark_t4.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import transformers  # noqa: E402

from kvengine import DEFAULT_MODEL, load_model  # noqa: E402
from kvengine.bench import (  # noqa: E402
    make_workload,
    run_hf_baseline,
    run_kvengine,
    workload_summary,
)

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def environment(device: torch.device, dtype: torch.dtype, model_id: str) -> dict:
    env = {
        "model": model_id,
        "device": str(device),
        "dtype": str(dtype),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_memory_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 2
        )
    return env


def fmt_pcts(d: dict) -> str:
    return f"p50 {d['p50']:.3f}  p90 {d['p90']:.3f}  p99 {d['p99']:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES))
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-mean", type=float, default=128)
    ap.add_argument("--output-mean", type=float, default=64)
    ap.add_argument("--blocks", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch-size", type=int, default=16)
    ap.add_argument("--baseline-batch-size", type=int, default=16)
    ap.add_argument("--no-sort-baseline", action="store_true",
                    help="disable length-sorted batching (makes the baseline weaker)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--out", default=None, help="path to write JSON results")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    model, tokenizer = load_model(args.model, device=args.device, dtype=dtype)
    device = next(model.parameters()).device
    if dtype is torch.float16 and device.type == "cpu":
        print("warning: float16 on CPU is not well supported and will be slow",
              file=sys.stderr)

    workload = make_workload(
        tokenizer,
        n_requests=args.requests,
        seed=args.seed,
        prompt_mean=args.prompt_mean,
        output_mean=args.output_mean,
    )
    wl = workload_summary(workload)

    env = environment(device, dtype, args.model)
    print(f"model {args.model} | device {device} | dtype {dtype}")
    if "gpu" in env:
        print(f"gpu   {env['gpu']} ({env['gpu_memory_gib']} GiB)")
    print(f"workload: {wl['n_requests']} requests, "
          f"{wl['prompt_tokens_total']} prompt tokens, "
          f"{wl['output_tokens_requested']} output tokens requested")
    print(f"  prompt len  {wl['prompt_len']}")
    print(f"  output len  {wl['output_len']}")
    print()

    print("running kvengine ...", flush=True)
    engine_result = run_kvengine(
        model,
        workload,
        num_blocks=args.blocks,
        block_size=args.block_size,
        max_batch_size=args.max_batch_size,
    )

    baseline_result = None
    if not args.skip_baseline:
        print("running huggingface baseline ...", flush=True)
        baseline_result = run_hf_baseline(
            model,
            workload,
            batch_size=args.baseline_batch_size,
            sort_by_length=not args.no_sort_baseline,
        )

    runs = [r for r in (baseline_result, engine_result) if r is not None]
    print()
    print("=" * 78)
    for r in runs:
        s = r.summary()
        print(f"\n{s['label']}")
        print(f"  wall             {s['wall_s']:.2f}s")
        print(f"  throughput       {s['output_tokens_per_s']:.1f} output tok/s "
              f"| {s['total_tokens_per_s']:.1f} total tok/s "
              f"| {s['requests_per_s']:.2f} req/s")
        print(f"  latency (s)      {fmt_pcts(s['latency_s'])}")
        print(f"  ttft (s)         {fmt_pcts(s['ttft_s'])}")
        print(f"  tpot (s/token)   {fmt_pcts(s['tpot_s'])}")
        cache = s.get("cache", {})
        print(f"  cache util       {cache.get('slot_utilization', 0):.1%} "
              f"of held slots hold real tokens")
        if "scheduler" in s:
            sched = s["scheduler"]
            print(f"  scheduler        avg decode batch {sched['avg_decode_batch']:.2f} "
                  f"(max {sched['max_decode_batch']}), "
                  f"{sched['preemptions']} preemptions, "
                  f"{sched['avg_padding_waste']:.1%} padding waste")

    if baseline_result is not None:
        b, e = baseline_result.summary(), engine_result.summary()
        speedup = e["output_tokens_per_s"] / b["output_tokens_per_s"] if b["output_tokens_per_s"] else 0
        util_gain = (
            e["cache"]["slot_utilization"] / b["cache"]["slot_utilization"]
            if b["cache"]["slot_utilization"]
            else 0
        )
        print()
        print("-" * 78)
        print(f"throughput:      {speedup:.2f}x")
        print(f"cache util:      {b['cache']['slot_utilization']:.1%} -> "
              f"{e['cache']['slot_utilization']:.1%} ({util_gain:.2f}x)")
        print(f"p90 latency:     {b['latency_s']['p90']:.2f}s -> "
              f"{e['latency_s']['p90']:.2f}s")
        print(f"p90 ttft:        {b['ttft_s']['p90']:.2f}s -> "
              f"{e['ttft_s']['p90']:.2f}s")

    if args.out:
        payload = {
            "environment": env,
            "config": vars(args),
            "workload": wl,
            "runs": {
                ("kvengine" if r is engine_result else "huggingface"): r.to_json()
                for r in runs
            },
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
