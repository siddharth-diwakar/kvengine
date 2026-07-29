"""Phase 5: speculative decoding.

The bottleneck this attacks: generating one token requires reading every weight of
the target model, so decode is memory-bandwidth-bound and the arithmetic units sit
mostly idle. Speculative decoding fills them. A small draft model proposes k tokens
cheaply, then the target model checks all k **in a single forward pass** — because
verifying k tokens is one pass over k positions, which costs barely more than one
pass over one position. Every accepted token is a token the target never had to
generate serially.

The guarantee, and it is exact
------------------------------
Greedy speculative decoding produces **byte-identical output to plain greedy
decoding of the target model**. The draft model affects only *speed*, never the
result: a token is accepted precisely when the target would have chosen it anyway,
and the first disagreement is resolved in the target's favour. A useless draft
model makes this slower than plain decoding; it cannot make it wrong. That is the
correctness anchor the tests use, and it holds for any draft, any k.

The subtle part is bookkeeping, not sampling
--------------------------------------------
The target writes K/V for every drafted token before it knows which are good, so
rejected tokens leave entries in the cache that must be rolled back — otherwise
the next pass attends to tokens that were never generated. Both caches also drift
out of sync with the accepted sequence at different rates, so each iteration starts
by feeding whichever tokens that cache has not seen yet. Getting this off by one
produces output that is subtly wrong rather than obviously broken.

Positions, precisely
--------------------
With `u` tokens the target has not yet seen and k drafts, the target is fed
`u + k` tokens. Logits at index i predict the token *after* input i, so:

    index u-1      predicts drafts[0]      (it is the last real token)
    index u-1+j    predicts drafts[j]
    index u-1+k    predicts the bonus token after drafts[k-1]

That is k+1 predictions from one pass: k to check, plus a free token if all k are
accepted. So an iteration yields between 1 and k+1 tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .decode import AUTO_EOS, _resolve_eos, _sync
from .forward import forward_with_own_cache


@dataclass
class SpecDecodeResult:
    prompt_token_ids: list[int]
    new_token_ids: list[int]
    elapsed_s: float
    stopped_on_eos: bool

    iterations: int = 0
    tokens_drafted: int = 0
    tokens_accepted: int = 0
    # Accepted count per iteration, in [0, k]. The distribution matters more than
    # the mean: a draft that is usually right and occasionally hopeless behaves
    # very differently from one that is uniformly mediocre.
    accepted_per_iter: list[int] = field(default_factory=list)
    target_forwards: int = 0
    draft_forwards: int = 0

    @property
    def tokens_per_s(self) -> float:
        return len(self.new_token_ids) / self.elapsed_s if self.elapsed_s else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens the target agreed with.

        The headline quality number for the draft model. Speedup tracks this, but
        not linearly: drafting is not free, so a low rate can be a net loss.
        """
        return self.tokens_accepted / self.tokens_drafted if self.tokens_drafted else 0.0

    @property
    def tokens_per_target_forward(self) -> float:
        """Tokens produced per target forward pass.

        The number that explains the speedup. Plain greedy decoding is exactly 1.0
        by definition, so anything above 1.0 is work the target did not do
        serially.
        """
        return (
            len(self.new_token_ids) / self.target_forwards if self.target_forwards else 0.0
        )

    def summary(self) -> dict:
        return {
            "new_tokens": len(self.new_token_ids),
            "elapsed_s": round(self.elapsed_s, 3),
            "tokens_per_s": round(self.tokens_per_s, 2),
            "iterations": self.iterations,
            "tokens_drafted": self.tokens_drafted,
            "tokens_accepted": self.tokens_accepted,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "tokens_per_target_forward": round(self.tokens_per_target_forward, 3),
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "stopped_on_eos": self.stopped_on_eos,
        }


@torch.no_grad()
def speculative_greedy(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    target_cache,
    draft_cache,
    k: int = 4,
    max_new_tokens: int = 32,
    eos_token_id=AUTO_EOS,
) -> SpecDecodeResult:
    """Greedy speculative decoding. Output is identical to greedy on the target.

    `target_cache` and `draft_cache` must be empty and must support truncate();
    either ContiguousKVCache or PagedSequenceCache works.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "one request at a time"
    assert k >= 1, "k must be at least 1"
    assert target_cache.length == 0 and draft_cache.length == 0, "caches must be empty"

    device = input_ids.device
    eos_ids = _resolve_eos(target_model, eos_token_id)

    prompt = input_ids[0].tolist()
    tokens = list(prompt)
    result = SpecDecodeResult(
        prompt_token_ids=prompt, new_token_ids=[], elapsed_s=0.0, stopped_on_eos=False
    )

    def feed(model, cache, ids: list[int]) -> torch.Tensor:
        batch = torch.tensor([ids], dtype=torch.long, device=device)
        return forward_with_own_cache(model, batch, cache, all_logits=True)[0]

    _sync(device)
    t0 = time.perf_counter()

    while len(result.new_token_ids) < max_new_tokens:
        # --- 1. draft proposes k tokens, one cheap forward each ---
        unseen = tokens[draft_cache.length :]
        logits = feed(draft_model, draft_cache, unseen)
        result.draft_forwards += 1

        drafts: list[int] = []
        for j in range(k):
            drafts.append(int(torch.argmax(logits[-1])))
            if j < k - 1:
                logits = feed(draft_model, draft_cache, [drafts[-1]])
                result.draft_forwards += 1

        # --- 2. target verifies all k in ONE forward pass ---
        unseen_t = tokens[target_cache.length :]
        u = len(unseen_t)
        t_logits = feed(target_model, target_cache, unseen_t + drafts)
        result.target_forwards += 1

        # index u-1+j predicts drafts[j]; index u-1+k is the bonus
        target_preds = torch.argmax(t_logits[u - 1 :], dim=-1).tolist()

        # --- 3. accept the longest matching prefix ---
        n_accepted = 0
        for j in range(k):
            if target_preds[j] != drafts[j]:
                break
            n_accepted += 1

        if n_accepted == k:
            # All agreed, so the bonus prediction is a free extra token.
            emitted = drafts + [target_preds[k]]
        else:
            # First disagreement resolved in the target's favour. This is what
            # makes the output exactly equal to greedy target decoding.
            emitted = drafts[:n_accepted] + [target_preds[n_accepted]]

        result.iterations += 1
        result.tokens_drafted += k
        result.tokens_accepted += n_accepted
        result.accepted_per_iter.append(n_accepted)

        # --- 4. commit, and roll both caches back to the accepted sequence ---
        tokens.extend(emitted)
        result.new_token_ids.extend(emitted)

        # Invariant for the next iteration: every token but the last has K/V, so
        # feeding the last one produces the next prediction.
        keep = len(tokens) - 1
        target_cache.truncate(min(target_cache.length, keep))
        draft_cache.truncate(min(draft_cache.length, keep))

        if any(t in eos_ids for t in emitted):
            first = next(i for i, t in enumerate(emitted) if t in eos_ids)
            # Drop everything after EOS; it was speculated past the end.
            trim = len(emitted) - (first + 1)
            if trim:
                result.new_token_ids = result.new_token_ids[:-trim]
            result.stopped_on_eos = True
            break

    _sync(device)
    result.elapsed_s = time.perf_counter() - t0

    # An iteration emits up to k+1 tokens, so the budget can overshoot.
    if len(result.new_token_ids) > max_new_tokens:
        result.new_token_ids = result.new_token_ids[:max_new_tokens]
        result.stopped_on_eos = result.new_token_ids[-1] in eos_ids

    return result
