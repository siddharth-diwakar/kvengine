"""Phase 3: batched decode over the paged pool.

Prefill and decode want opposite things from a batch, and that shapes the whole
design:

  prefill  processes a whole prompt at once, so a single request already gives
           the matmuls hundreds of rows of work. It is compute-bound.
  decode   processes one token per request. Alone, a decode step reads all the
           model weights to do a single token of work. It is memory-bound, and
           the only way to make it efficient is to batch many requests into one
           pass so the weight read is amortised.

So this module batches decode and leaves prefill per-request. See the scheduler
in engine.py for the policy discussion.

The awkward part: batched decode has one query token per request but a different
history length per request. This implementation pads the gathered K/V to the
longest history in the batch and masks the padding out. Real vLLM instead uses
varlen kernels that consume a ragged batch with no padding at all — padding
wastes work proportional to the spread of lengths in the batch, which is why
production schedulers try to group requests of similar length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .forward import attend, apply_rope
from .paged import PagedSequenceCache


@dataclass
class DecodeBatch:
    """Everything one batched decode step needs, computed once for all layers.

    Building this per step rather than per layer matters: the index arithmetic
    would otherwise run 24 times for the same answer.
    """

    sequences: list[PagedSequenceCache]
    tokens: torch.Tensor      # [B, 1]   the token each request feeds in
    positions: torch.Tensor   # [B, 1]   absolute position of that token
    write_slots: torch.Tensor # [B]      physical slot it is written to
    read_slots: torch.Tensor  # [B, max_len] physical slot per history position
    mask: torch.Tensor        # [B, 1, 1, max_len] additive attention mask
    # History length per request including this step's token. Recorded at plan
    # time rather than read back off the sequences, because forward_decode_batch
    # commits and would otherwise make this depend on when it is called.
    lengths: list[int] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.sequences)

    @property
    def max_len(self) -> int:
        return self.read_slots.shape[1]

    def padding_waste(self) -> float:
        """Fraction of gathered K/V that is padding.

        The cost of padding instead of using a varlen kernel, and the number that
        justifies grouping requests of similar length in a batch.
        """
        total = self.batch_size * self.max_len
        return 1.0 - sum(self.lengths) / total if total else 0.0


def plan_decode_batch(
    sequences: list[PagedSequenceCache], next_tokens: list[int]
) -> DecodeBatch:
    """Reserve one slot per request and precompute every index the step needs.

    Callers must be ready for OutOfBlocks: a decode that cannot grow is what
    forces the scheduler to preempt. Reservation happens here, before any tensor
    work, so a failure leaves nothing half-done.
    """
    assert len(sequences) == len(next_tokens), "one token per sequence"
    assert sequences, "empty decode batch"
    device = sequences[0].pool.device

    # Lengths BEFORE this token, which are both the RoPE positions and the
    # number of valid history entries.
    positions = [s.length for s in sequences]

    write_slots = []
    for seq in sequences:
        seq.reserve(1)  # may raise OutOfBlocks; nothing mutated on failure
        write_slots.append(seq.slot_indices(seq.length, 1))

    # History AFTER this token is written: each request attends to itself too.
    lengths = [p + 1 for p in positions]
    max_len = max(lengths)

    read_slots = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    valid = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        read_slots[i, :length] = seq.slot_indices(0, length)
        valid[i, :length] = True

    # Padding slots point at slot 0, which holds some other request's data. The
    # mask is what makes that safe, so it is not optional.
    mask = torch.zeros((len(sequences), 1, 1, max_len), dtype=sequences[0].pool.dtype, device=device)
    mask.masked_fill_(~valid[:, None, None, :], torch.finfo(mask.dtype).min)

    return DecodeBatch(
        sequences=sequences,
        tokens=torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1),
        positions=torch.tensor(positions, dtype=torch.long, device=device).unsqueeze(1),
        write_slots=torch.cat(write_slots),
        read_slots=read_slots,
        mask=mask,
        lengths=lengths,
    )


@torch.no_grad()
def forward_decode_batch(model, batch: DecodeBatch) -> torch.Tensor:
    """One decode step for the whole batch. Returns logits [B, vocab].

    Same layer structure as the single-request forward pass. The differences are
    all in the cache access: a scattered write of B tokens, and a padded gather
    of B histories.
    """
    inner = model.model
    pool = batch.sequences[0].pool
    n_batch = batch.batch_size
    max_len = batch.max_len

    hidden = inner.embed_tokens(batch.tokens)  # [B, 1, hidden]
    # Each request sits at its own absolute position, so RoPE differs per row.
    cos, sin = inner.rotary_emb(hidden, batch.positions)

    for layer in inner.layers:
        attn = layer.self_attn
        residual = hidden
        h = layer.input_layernorm(hidden)

        qkv_shape = (n_batch, 1, -1, attn.head_dim)
        q = attn.q_proj(h).view(qkv_shape).transpose(1, 2)
        k = attn.k_proj(h).view(qkv_shape).transpose(1, 2)
        v = attn.v_proj(h).view(qkv_shape).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        # Scatter: one token per request, each to its own physical slot.
        pool.k_flat[attn.layer_idx].index_copy_(0, batch.write_slots, k[:, :, 0, :])
        pool.v_flat[attn.layer_idx].index_copy_(0, batch.write_slots, v[:, :, 0, :])

        # Gather: one strided read for the whole batch, padded to max_len.
        flat_slots = batch.read_slots.reshape(-1)
        k_all = pool.k_flat[attn.layer_idx][flat_slots]
        v_all = pool.v_flat[attn.layer_idx][flat_slots]
        shape = (n_batch, max_len, pool.num_kv_heads, pool.head_dim)
        k_all = k_all.reshape(shape).transpose(1, 2)
        v_all = v_all.reshape(shape).transpose(1, 2)

        attn_out = attend(attn, q, k_all, v_all, batch.mask)
        hidden = residual + attn.o_proj(attn_out.reshape(n_batch, 1, -1))

        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    for seq in batch.sequences:
        seq.commit(1)

    hidden = inner.norm(hidden)
    return model.lm_head(hidden)[:, -1, :]
