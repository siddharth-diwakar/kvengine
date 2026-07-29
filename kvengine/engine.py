"""Phase 3: continuous batching.

A waiting queue, a running batch, and a scheduler loop that admits requests when
blocks are available and retires them the moment they finish. "Continuous" is the
whole point: a finished request's blocks are returned and its slot in the batch
refilled on the very next step, rather than the batch running to the length of
its slowest member.

The design decision worth writing up
------------------------------------
New requests need a prefill pass over the whole prompt; running requests need a
one-token decode pass. They cannot share a forward pass without extra machinery,
so the scheduler must choose an order, and the choice is a real tradeoff:

  prefill_first (default)
      Admit and prefill as soon as blocks allow. Minimises time-to-first-token,
      but every prefill stalls all running requests for a step, so inter-token
      latency gets spiky under load. `max_prefills_per_step` caps the damage.

  decode_first
      Only prefill when nothing is decoding. Smooth inter-token latency and the
      best decode throughput, at the cost of new arrivals waiting behind long
      generations.

Production engines dodge the dilemma with *chunked prefill*: split a long prompt
into fixed-size pieces and mix one piece into a decode batch, so neither side
stalls. The correctness prerequisite for that already passes here
(`test_incremental_prefill_equals_single_prefill` in phase 1); what is missing is
a forward pass that takes ragged prefill chunks and decode tokens together, which
is the natural next step.

Preemption
----------
A decode step can fail to allocate: every running request wants one more slot and
the pool is full. The scheduler then *preempts* the most recently admitted request
— frees its blocks and returns it to the waiting queue — so older requests keep
making progress (FCFS fairness). Preempted requests are resumed by recomputation:
their generated tokens are kept, and on re-admission the prompt plus those tokens
are prefilled again to rebuild the cache. vLLM calls this recompute preemption and
does exactly the same thing (its alternative is swapping blocks to host memory).
Recomputation costs a prefill but needs no extra memory, which is the right trade
when memory is precisely what ran out.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch

from .batch import forward_decode_batch, plan_decode_batch
from .blocks import OutOfBlocks
from .decode import AUTO_EOS, _resolve_eos
from .forward import forward_with_own_cache
from .paged import PagedKVCache


class RequestState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    """One in-flight generation request."""

    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    eos_ids: set[int] = field(default_factory=set)

    state: RequestState = RequestState.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    seq: object | None = None
    finish_reason: str | None = None

    # metrics, in scheduler steps
    arrival_step: int = 0
    first_token_step: int | None = None
    finish_step: int | None = None
    preemptions: int = 0
    prefill_tokens: int = 0  # includes tokens recomputed after preemption

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def next_token(self) -> int:
        """The token to feed on the next decode step."""
        return self.output_token_ids[-1]

    def prefill_input(self) -> list[int]:
        """Tokens to prefill so the cache is correct and next_token is unconsumed.

        Fresh request: the prompt. Resumed after preemption: the prompt plus every
        generated token except the last, because the last one has not been fed to
        the model yet. Getting this boundary wrong duplicates or drops a token.
        """
        if not self.output_token_ids:
            return self.prompt_token_ids
        return self.prompt_token_ids + self.output_token_ids[:-1]

    @property
    def is_resumed(self) -> bool:
        return bool(self.output_token_ids)

    def is_done(self) -> bool:
        if len(self.output_token_ids) >= self.max_new_tokens:
            return True
        return bool(self.output_token_ids) and self.output_token_ids[-1] in self.eos_ids

    def done_reason(self) -> str:
        if self.output_token_ids and self.output_token_ids[-1] in self.eos_ids:
            return "eos"
        return "length"


@dataclass
class StepInfo:
    """What one scheduler step did. The raw material for the phase 4 benchmark."""

    step: int
    kind: str  # "prefill" | "decode" | "idle"
    batch_size: int = 0
    tokens_generated: int = 0
    prefilled: list[str] = field(default_factory=list)
    admitted: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)
    aborted: list[str] = field(default_factory=list)
    blocks_in_use: int = 0
    pool_utilization: float = 0.0
    slot_utilization: float = 0.0
    padding_waste: float = 0.0


class Engine:
    """Continuous-batching engine over a paged KV pool."""

    def __init__(
        self,
        model,
        pool: PagedKVCache,
        max_batch_size: int = 8,
        max_prefills_per_step: int = 1,
        prefill_first: bool = True,
        watermark_blocks: int = 1,
    ):
        self.model = model
        self.pool = pool
        self.max_batch_size = max_batch_size
        self.max_prefills_per_step = max_prefills_per_step
        self.prefill_first = prefill_first
        # Headroom kept free at admission so a freshly admitted request can grow
        # for at least one step before it triggers a preemption. Admitting right
        # up to the last block guarantees thrashing.
        self.watermark_blocks = watermark_blocks

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.step_count = 0
        self.history: list[StepInfo] = []

    # --- submission ---

    def add_request(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int = 32,
        request_id: str | None = None,
        eos_token_id=AUTO_EOS,
    ) -> Request:
        req = Request(
            request_id=request_id or f"req-{len(self.waiting) + len(self.running) + len(self.finished)}",
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            eos_ids=_resolve_eos(self.model, eos_token_id),
            arrival_step=self.step_count,
        )
        self.waiting.append(req)
        return req

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # --- the scheduler loop ---

    def step(self) -> StepInfo:
        """Run one scheduler iteration: admit, then either prefill or decode."""
        info = StepInfo(step=self.step_count, kind="idle")

        self._admit(info)
        # Derived from state, not from what _admit just returned: a request
        # admitted on an earlier step but skipped by a decode-first policy must
        # still get prefilled, not stranded in the running list forever.
        pending_prefill = self._needs_prefill()

        if pending_prefill and (self.prefill_first or not self._decodable()):
            info.kind = "prefill"
            self._run_prefills(pending_prefill, info)
        elif self._decodable():
            info.kind = "decode"
            self._run_decode(info)
        elif pending_prefill:
            # decode_first policy with nothing decodable yet
            info.kind = "prefill"
            self._run_prefills(pending_prefill, info)

        if info.kind == "idle" and not self.running and self.waiting:
            # Backstop: nothing running, nothing admissible, so no future step can
            # differ from this one. Abort the head of the queue rather than spin.
            stuck = self.waiting.popleft()
            self._retire(stuck, "aborted_no_progress", info)
            info.aborted.append(stuck.request_id)

        self._record(info)
        self.step_count += 1
        self.history.append(info)
        return info

    def run(self, max_steps: int = 100_000) -> list[Request]:
        """Drive the loop until every request is finished."""
        while self.has_work:
            if self.step_count >= max_steps:
                raise RuntimeError(f"scheduler did not converge in {max_steps} steps")
            self.step()
        return self.finished

    # --- admission ---

    def _admit(self, info: StepInfo) -> list[Request]:
        """Move requests from waiting to running while blocks allow.

        Blocks are allocated lazily by the forward pass, so admission only needs
        to check that the prompt would fit with headroom to spare.

        Admission is paced at max_prefills_per_step because an admitted request
        is prefilled before it can decode; admitting faster than we prefill just
        builds a backlog inside the running list.
        """
        admitted: list[Request] = []
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting[0]
            needed = self.pool.manager.blocks_needed(len(req.prefill_input()))

            if needed + self.watermark_blocks > self.pool.num_blocks:
                # Would not fit even in a completely empty pool, so waiting can
                # never help. Without this the scheduler spins forever on a
                # request it can never admit.
                self.waiting.popleft()
                self._retire(req, "aborted_prompt_too_long", info)
                info.aborted.append(req.request_id)
                continue

            if needed + self.watermark_blocks > self.pool.manager.num_free:
                break
            self.waiting.popleft()
            req.state = RequestState.RUNNING
            req.seq = self.pool.new_sequence(req.request_id)
            self.running.append(req)
            admitted.append(req)
            info.admitted.append(req.request_id)
            if len(admitted) >= self.max_prefills_per_step:
                break
        return admitted

    def _decodable(self) -> list[Request]:
        """Running requests that have been prefilled and are ready for a decode."""
        return [r for r in self.running if r.seq is not None and r.seq.length > 0]

    def _needs_prefill(self) -> list[Request]:
        """Running requests whose cache is still empty."""
        pending = [r for r in self.running if r.seq is not None and r.seq.length == 0]
        return pending[: self.max_prefills_per_step]

    # --- prefill ---

    def _run_prefills(self, requests: list[Request], info: StepInfo) -> None:
        """Prefill each newly admitted request, one forward pass per prompt.

        A prompt already gives the matmuls plenty of rows, so batching prompts
        together buys much less than batching decodes does. Keeping prefill
        per-request also keeps the ragged-length problem out of this code path.
        """
        for req in requests:
            tokens = req.prefill_input()
            ids = torch.tensor([tokens], dtype=torch.long, device=self.pool.device)
            try:
                logits = forward_with_own_cache(self.model, ids, req.seq)
            except OutOfBlocks:
                # Admission checked capacity, so this means the pool is genuinely
                # too small for this prompt. Put it back and let the scheduler
                # retry once other requests retire.
                req.seq.free()
                req.seq = None
                req.state = RequestState.WAITING
                self.running.remove(req)
                self.waiting.appendleft(req)
                continue

            req.prefill_tokens += len(tokens)
            info.prefilled.append(req.request_id)

            if not req.is_resumed:
                token = int(torch.argmax(logits[0, -1, :]))
                req.output_token_ids.append(token)
                req.first_token_step = self.step_count
                info.tokens_generated += 1
                if req.is_done():
                    self._retire(req, req.done_reason(), info)

    # --- decode ---

    def _run_decode(self, info: StepInfo) -> None:
        """One batched decode step, preempting if the pool cannot grow."""
        while True:
            active = self._decodable()
            if not active:
                info.kind = "idle"
                return
            try:
                batch = plan_decode_batch(
                    [r.seq for r in active], [r.next_token for r in active]
                )
                break
            except OutOfBlocks:
                if len(active) == 1:
                    # Nothing left to preempt: this request cannot be served by a
                    # pool this small. Abort it explicitly rather than truncate
                    # its output and pretend it finished normally.
                    self._retire(active[0], "aborted_cache_exhausted", info)
                    info.aborted.append(active[0].request_id)
                    info.kind = "idle"
                    return
                self._preempt(active[-1], info)
                # Retry: reserve() only allocates the blocks a sequence still
                # lacks, so partially-reserved sequences from the failed attempt
                # are fine to re-plan.

        logits = forward_decode_batch(self.model, batch)
        info.batch_size = batch.batch_size
        info.padding_waste = batch.padding_waste()

        next_ids = torch.argmax(logits, dim=-1).tolist()
        for req, token in zip(active, next_ids):
            req.output_token_ids.append(int(token))
            info.tokens_generated += 1
            if req.is_done():
                self._retire(req, req.done_reason(), info)

    # --- state transitions ---

    def _preempt(self, req: Request, info: StepInfo) -> None:
        """Free a running request's blocks and return it to the queue.

        Generated tokens are kept, so no output is lost — only the cache is, and
        it is rebuilt by recomputation on re-admission.
        """
        req.seq.free()
        req.seq = None
        req.state = RequestState.WAITING
        req.preemptions += 1
        self.running.remove(req)
        # Front of the queue: it is the oldest work among the waiting, so it
        # should resume before anything that has not started.
        self.waiting.appendleft(req)
        info.preempted.append(req.request_id)

    def _retire(self, req: Request, reason: str, info: StepInfo) -> None:
        """Release blocks immediately. This is what makes the batching continuous."""
        if req.seq is not None:
            req.seq.free()
            req.seq = None
        req.state = RequestState.FINISHED
        req.finish_reason = reason
        req.finish_step = self.step_count
        if req in self.running:
            self.running.remove(req)
        self.finished.append(req)
        info.finished.append(req.request_id)

    # --- metrics ---

    def _record(self, info: StepInfo) -> None:
        mgr = self.pool.manager
        info.blocks_in_use = mgr.num_allocated
        info.pool_utilization = mgr.pool_utilization()
        held = sum(r.seq.capacity for r in self.running if r.seq is not None)
        real = sum(r.seq.length for r in self.running if r.seq is not None)
        info.slot_utilization = real / held if held else 0.0

    def stats(self) -> dict:
        decode_steps = [s for s in self.history if s.kind == "decode"]
        prefill_steps = [s for s in self.history if s.kind == "prefill"]
        generated = sum(s.tokens_generated for s in self.history)
        return {
            "steps": self.step_count,
            "prefill_steps": len(prefill_steps),
            "decode_steps": len(decode_steps),
            "idle_steps": sum(1 for s in self.history if s.kind == "idle"),
            "requests_finished": len(self.finished),
            "tokens_generated": generated,
            "preemptions": sum(r.preemptions for r in self.finished),
            "aborted": sum(1 for r in self.finished if r.finish_reason.startswith("aborted")),
            "avg_decode_batch": (
                sum(s.batch_size for s in decode_steps) / len(decode_steps)
                if decode_steps
                else 0.0
            ),
            "max_decode_batch": max((s.batch_size for s in decode_steps), default=0),
            "peak_pool_utilization": max((s.pool_utilization for s in self.history), default=0.0),
            "avg_slot_utilization": (
                sum(s.slot_utilization for s in self.history) / len(self.history)
                if self.history
                else 0.0
            ),
            "avg_padding_waste": (
                sum(s.padding_waste for s in decode_steps) / len(decode_steps)
                if decode_steps
                else 0.0
            ),
            "prefill_tokens": sum(r.prefill_tokens for r in self.finished),
        }
