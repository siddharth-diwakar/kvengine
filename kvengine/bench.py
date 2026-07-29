"""Phase 4: benchmark harness.

Three pieces: a reproducible workload with realistic length spread, a tuned
HuggingFace baseline to beat, and the metric computation shared by both so the
numbers are defined identically on each side.

What makes a baseline "tuned" here
----------------------------------
Comparing against naive one-at-a-time `.generate()` would be a strawman. The
baseline therefore uses fp16, batches requests, and sorts by prompt length so
each batch pads to a similar size. What it cannot do is refill a slot when a
request finishes early — a static batch runs until its longest member is done.
That is exactly the gap continuous batching closes, so it should be visible in
the numbers rather than assumed.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field

import torch

from .engine import Engine
from .paged import PagedKVCache
from .reference import strict_greedy_config

# Source text for prompt construction. Prompts are windows of real tokenized text
# rather than random token ids, so prompt lengths and the tokenizer's behaviour are
# both realistic. Content is irrelevant to throughput; realism of *length* is not.
CORPUS = """
Attention mechanisms let a model weigh every earlier position when producing the
next one. In an autoregressive decoder, each new token attends over the keys and
values of all previous tokens, which means those keys and values are computed once
and reused for the remainder of the sequence. Caching them turns a quadratic
amount of recomputation into a linear amount of memory, and that memory quickly
becomes the binding constraint on how many requests a server can hold at once.
The size of the cache grows with the number of layers, the number of key value
heads, the head dimension, and the length of the sequence. Serving many requests
concurrently therefore becomes a memory allocation problem rather than a pure
compute problem. A naive allocator reserves the maximum possible length for every
request before it knows how long the answer will be, which wastes most of the
memory it reserves. Paging the cache into fixed size blocks and handing them out
on demand bounds that waste to less than one block per request. Once memory is no
longer the bottleneck, the scheduler decides throughput: it must admit new work
when blocks are free, retire finished work immediately, and choose how to mix
prompt processing with token generation. Prompt processing is compute bound
because a single prompt already provides many rows of work to the matrix
multiplications. Token generation is memory bandwidth bound because a single
token per request still requires reading every weight in the model. Batching many
requests into one generation step amortises that read, which is why throughput
improves even though each individual request does not get faster. Speculative
decoding attacks the same bottleneck from another direction by proposing several
tokens with a small draft model and verifying them in one pass of the large model,
accepting the longest prefix the large model agrees with.
"""


@dataclass
class WorkloadRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


def _lognormal_length(rng: random.Random, mean: float, sigma: float, lo: int, hi: int) -> int:
    """Sample a length from a lognormal, clamped.

    Lognormal because real request lengths are right-skewed: most are short, a few
    are very long. A uniform distribution would hide exactly the behaviour we care
    about, since padding waste and preemption are both driven by the spread.
    """
    mu = math.log(max(mean, 1.0))
    return max(lo, min(hi, int(round(math.exp(rng.gauss(mu, sigma))))))


def make_workload(
    tokenizer,
    n_requests: int,
    seed: int = 0,
    prompt_mean: float = 128,
    prompt_sigma: float = 0.6,
    output_mean: float = 64,
    output_sigma: float = 0.6,
    min_prompt: int = 8,
    max_prompt: int = 512,
    min_output: int = 4,
    max_output: int = 256,
) -> list[WorkloadRequest]:
    """Build a deterministic workload with right-skewed prompt and output lengths."""
    rng = random.Random(seed)
    corpus_ids = tokenizer(CORPUS.strip(), return_tensors=None)["input_ids"]

    requests = []
    for i in range(n_requests):
        p_len = _lognormal_length(rng, prompt_mean, prompt_sigma, min_prompt, max_prompt)
        o_len = _lognormal_length(rng, output_mean, output_sigma, min_output, max_output)

        # Tile the corpus so prompts longer than it are still exactly p_len tokens.
        reps = -(-p_len // len(corpus_ids)) + 1
        tiled = corpus_ids * reps
        start = rng.randrange(len(corpus_ids))
        prompt_ids = tiled[start : start + p_len]

        requests.append(
            WorkloadRequest(f"r{i}", list(prompt_ids), max_new_tokens=o_len)
        )
    return requests


def workload_summary(workload: list[WorkloadRequest]) -> dict:
    prompts = [r.prompt_len for r in workload]
    outputs = [r.max_new_tokens for r in workload]
    return {
        "n_requests": len(workload),
        "prompt_tokens_total": sum(prompts),
        "output_tokens_requested": sum(outputs),
        "prompt_len": _describe(prompts),
        "output_len": _describe(outputs),
    }


def _describe(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": _pct(ordered, 50),
        "mean": round(statistics.fmean(ordered), 1),
        "p90": _pct(ordered, 90),
        "max": ordered[-1],
    }


def _pct(ordered: list[float], p: float) -> float:
    """Nearest-rank percentile. Explicit so both sides compute it identically."""
    if not ordered:
        return 0.0
    k = max(0, min(len(ordered) - 1, int(math.ceil(p / 100 * len(ordered))) - 1))
    return round(ordered[k], 4)


# --- per-request records, shared shape across both systems ---


@dataclass
class RequestRecord:
    request_id: str
    prompt_len: int
    output_len: int
    ttft_s: float | None
    latency_s: float | None
    tpot_s: float | None
    finish_reason: str
    preemptions: int = 0


@dataclass
class RunResult:
    label: str
    wall_s: float
    records: list[RequestRecord]
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        out_tokens = sum(r.output_len for r in self.records)
        prompt_tokens = sum(r.prompt_len for r in self.records)
        lat = sorted(r.latency_s for r in self.records if r.latency_s is not None)
        ttft = sorted(r.ttft_s for r in self.records if r.ttft_s is not None)
        tpot = sorted(r.tpot_s for r in self.records if r.tpot_s is not None)
        return {
            "label": self.label,
            "wall_s": round(self.wall_s, 3),
            "requests": len(self.records),
            "output_tokens": out_tokens,
            "prompt_tokens": prompt_tokens,
            "output_tokens_per_s": round(out_tokens / self.wall_s, 2) if self.wall_s else 0,
            "total_tokens_per_s": round(
                (out_tokens + prompt_tokens) / self.wall_s, 2
            ) if self.wall_s else 0,
            "requests_per_s": round(len(self.records) / self.wall_s, 3) if self.wall_s else 0,
            "latency_s": {"p50": _pct(lat, 50), "p90": _pct(lat, 90), "p99": _pct(lat, 99),
                          "mean": round(statistics.fmean(lat), 4) if lat else 0},
            "ttft_s": {"p50": _pct(ttft, 50), "p90": _pct(ttft, 90), "p99": _pct(ttft, 99),
                       "mean": round(statistics.fmean(ttft), 4) if ttft else 0},
            "tpot_s": {"p50": _pct(tpot, 50), "p90": _pct(tpot, 90), "p99": _pct(tpot, 99),
                       "mean": round(statistics.fmean(tpot), 4) if tpot else 0},
            **self.extra,
        }

    def to_json(self) -> dict:
        return {"summary": self.summary(), "requests": [asdict(r) for r in self.records]}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# --- our engine ---


def run_kvengine(
    model,
    workload: list[WorkloadRequest],
    num_blocks: int,
    block_size: int = 16,
    max_batch_size: int = 16,
    max_prefills_per_step: int = 1,
    prefill_first: bool = True,
) -> RunResult:
    pool = PagedKVCache.for_model(model, num_blocks=num_blocks, block_size=block_size)
    engine = Engine(
        model,
        pool,
        max_batch_size=max_batch_size,
        max_prefills_per_step=max_prefills_per_step,
        prefill_first=prefill_first,
    )
    device = pool.device

    _sync(device)
    t0 = time.perf_counter()
    for r in workload:
        engine.add_request(
            r.prompt_token_ids, max_new_tokens=r.max_new_tokens, request_id=r.request_id
        )
    finished = engine.run()
    _sync(device)
    wall = time.perf_counter() - t0

    records = [
        RequestRecord(
            request_id=r.request_id,
            prompt_len=r.prompt_len,
            output_len=len(r.output_token_ids),
            ttft_s=r.ttft_s,
            latency_s=r.latency_s,
            tpot_s=r.tpot_s,
            finish_reason=r.finish_reason,
            preemptions=r.preemptions,
        )
        for r in finished
    ]

    stats = engine.stats()
    return RunResult(
        label="kvengine (paged + continuous batching)",
        wall_s=wall,
        records=records,
        extra={
            "scheduler": stats,
            "cache": {
                # The headline memory number: of the slots requests actually held,
                # how many carried real tokens. Waste is bounded by one block each.
                "slot_utilization": round(stats["avg_slot_utilization"], 4),
                "peak_pool_utilization": round(stats["peak_pool_utilization"], 4),
                "num_blocks": num_blocks,
                "block_size": block_size,
                "pool_mib": round(pool.nbytes() / 2**20, 2),
            },
        },
    )


# --- tuned HuggingFace baseline ---


class _FirstTokenTimer:
    """StoppingCriteria that never stops, used only to observe the first token.

    `.generate()` gives no callback for token emission, but stopping criteria run
    after every step, so the first invocation is a good proxy for first-token time.
    Without this the baseline would have no TTFT at all to compare.
    """

    def __init__(self):
        self.first_call: float | None = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.first_call is None:
            self.first_call = time.perf_counter()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def run_hf_baseline(
    model,
    workload: list[WorkloadRequest],
    batch_size: int = 16,
    sort_by_length: bool = True,
    pad_token_id: int | None = None,
) -> RunResult:
    """Static batched `.generate()`, length-sorted and left-padded.

    Static batching means every request in a batch finishes when the longest one
    does, so a short request's cache sits reserved and idle until then. Its output
    is truncated to its own budget afterwards, which is what a real deployment
    would do, but the *time* was already spent.
    """
    device = next(model.parameters()).device
    if pad_token_id is None:
        eos = model.generation_config.eos_token_id
        pad_token_id = eos if isinstance(eos, int) else eos[0]

    order = sorted(workload, key=lambda r: r.prompt_len) if sort_by_length else list(workload)
    records: list[RequestRecord] = []
    cache_real = 0
    cache_allocated = 0
    prompt_pad_tokens = 0

    _sync(device)
    t0 = time.perf_counter()

    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        max_prompt = max(r.prompt_len for r in chunk)
        max_new = max(r.max_new_tokens for r in chunk)

        # Left padding, so every sequence's last real token is at the same index
        # and generation continues from it.
        input_ids = torch.full(
            (len(chunk), max_prompt), pad_token_id, dtype=torch.long, device=device
        )
        attn = torch.zeros((len(chunk), max_prompt), dtype=torch.long, device=device)
        for i, r in enumerate(chunk):
            input_ids[i, max_prompt - r.prompt_len :] = torch.tensor(
                r.prompt_token_ids, dtype=torch.long, device=device
            )
            attn[i, max_prompt - r.prompt_len :] = 1
            prompt_pad_tokens += max_prompt - r.prompt_len

        cfg = strict_greedy_config(model, max_new)
        cfg.pad_token_id = pad_token_id
        timer = _FirstTokenTimer()

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            generation_config=cfg,
            stopping_criteria=[timer],
        )
        _sync(device)
        batch_end = time.perf_counter()

        generated = out[:, max_prompt:]
        eos_ids = cfg.eos_token_id
        eos_set = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)

        for i, r in enumerate(chunk):
            toks = generated[i].tolist()
            # Truncate to this request's own budget, and stop at EOS.
            toks = toks[: r.max_new_tokens]
            reason = "length"
            for j, t in enumerate(toks):
                if t in eos_set:
                    toks = toks[: j + 1]
                    reason = "eos"
                    break
            n_out = len(toks)
            # Timed from t0, not from this batch's start. Every request arrives at
            # t0 in an offline benchmark, so a request sitting in batch 3 really has
            # been waiting since t0. Measuring from batch start would hide the
            # baseline's queueing delay while the engine reports its own, which
            # would make the latency comparison meaningless.
            # Static batching also means the request is only released when the
            # whole batch ends, however early it personally finished.
            records.append(
                RequestRecord(
                    request_id=r.request_id,
                    prompt_len=r.prompt_len,
                    output_len=n_out,
                    ttft_s=(timer.first_call - t0) if timer.first_call else None,
                    latency_s=batch_end - t0,
                    tpot_s=(
                        (batch_end - timer.first_call) / (n_out - 1)
                        if timer.first_call and n_out > 1
                        else None
                    ),
                    finish_reason=reason,
                )
            )
            cache_real += r.prompt_len + n_out

        # The cache HF actually held for this batch: padded prompt plus every
        # generated position, for every row, until the batch ended.
        cache_allocated += len(chunk) * (max_prompt + generated.shape[1])

    _sync(device)
    wall = time.perf_counter() - t0

    return RunResult(
        label="huggingface (fp16-capable, static batching, length-sorted)",
        wall_s=wall,
        records=records,
        extra={
            "cache": {
                "slot_utilization": round(cache_real / cache_allocated, 4)
                if cache_allocated
                else 0.0,
                "slots_allocated": cache_allocated,
                "slots_real": cache_real,
                "prompt_padding_tokens": prompt_pad_tokens,
            },
            "batch_size": batch_size,
            "sorted_by_length": sort_by_length,
        },
    )
