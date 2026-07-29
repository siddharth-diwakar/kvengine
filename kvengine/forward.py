"""Phase 1: our own forward pass, driving our own KV cache.

We reuse the model's learned pieces (embeddings, projections, layernorms, MLP,
rotary table) but hand-write everything that touches the cache: rotary
application, the K/V write, the attention math, and the causal mask. That split
is deliberate. Reimplementing linear layers would teach nothing and risk
numerical drift; owning the attention core is the entire point, and it is what
Phase 2 rewrites to gather from scattered blocks.

Numerics are matched to HuggingFace's eager attention on purpose: softmax in
float32 then cast back, and a large-negative additive mask rather than -inf.
Small deviations here are what make a correct cache look broken.
"""

from __future__ import annotations

from typing import Protocol

import os

import torch

from .cache import ContiguousKVCache

# Use PyTorch's fused attention kernel by default. Set KVENGINE_EAGER_ATTN=1 to
# force the spelled-out path, which the tests use to prove the two agree.
USE_SDPA = os.environ.get("KVENGINE_EAGER_ATTN", "0") != "1"


class KVCacheLike(Protocol):
    """What the forward pass needs from a cache. Both phase 1 and phase 2 satisfy it.

    The ordering contract is the important part: reserve() once, then write()
    once per layer at the same `start`, then commit() once.
    """

    @property
    def length(self) -> int: ...

    def reserve(self, n_tokens: int) -> None: ...

    def write(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start: int
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def commit(self, n_tokens: int) -> None: ...


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotary embedding's pairing trick: treat the head_dim as two halves."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k by their absolute position.

    cos/sin arrive as [B, T, head_dim] and unsqueeze to [B, 1, T, head_dim] so
    they broadcast across heads.

    This is why positions must be tracked precisely: RoPE bakes the position
    into the key at write time. A key written at the wrong position is silently
    wrong forever, which is the single easiest way to break a hand-rolled cache.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for grouped-query attention.

    [B, n_kv_heads, T, D] -> [B, n_kv_heads * n_rep, T, D], with each KV head
    duplicated to serve its group of query heads. expand() is a view, so this
    costs nothing until reshape() forces the copy.
    """
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def causal_mask(
    q_len: int, kv_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    """Additive mask forbidding a query from seeing keys after it.

    The queries in this pass occupy the LAST q_len positions of the kv_len
    history, so query i sits at absolute position kv_len - q_len + i. Off-by-one
    here is the classic prefill bug.

    A single decode token may attend to the entire history, so no mask is needed
    and None skips the add entirely.
    """
    if q_len == 1:
        return None
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask.masked_fill_(k_pos > q_pos, torch.finfo(dtype).min)
    return mask[None, None, :, :]


def attend(
    attn_module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    use_sdpa: bool | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention over gathered K/V.

    q: [B, n_q_heads, T, D];  k/v: [B, n_kv_heads, kv_len, D]
    returns [B, T, n_q_heads, D]

    Two implementations of the same math. The eager one spells out every step and
    is the reference; the SDPA one dispatches to PyTorch's fused kernel.

    The fused path matters for honest benchmarking, not just speed: HuggingFace's
    baseline uses SDPA, so measuring hand-rolled eager attention against it would
    conflate "paging costs throughput" with "my attention core is unoptimised".
    Paging lives in how K/V is *gathered*, which is unchanged either way.
    """
    if use_sdpa is None:
        use_sdpa = USE_SDPA

    k = repeat_kv(k, attn_module.num_key_value_groups)
    v = repeat_kv(v, attn_module.num_key_value_groups)

    if use_sdpa:
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=attn_module.scaling
        )
        return out.transpose(1, 2).contiguous()

    scores = torch.matmul(q, k.transpose(2, 3)) * attn_module.scaling
    if mask is not None:
        scores = scores + mask
    # float32 softmax then cast back, matching HF's eager path exactly.
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v).transpose(1, 2).contiguous()


@torch.no_grad()
def forward_with_own_cache(
    model,
    input_ids: torch.Tensor,
    cache: "ContiguousKVCache | KVCacheLike",
    all_logits: bool = False,
) -> torch.Tensor:
    """One forward pass over `input_ids`, appending to `cache`.

    Handles prefill (T tokens, empty cache) and decode (1 token, warm cache)
    with the same code path — the only difference is T and the cache offset.

    `cache` is anything implementing length/reserve/write/commit, so this drives
    the phase 1 contiguous cache and the phase 2 paged cache without changes.
    Paging is entirely a property of how write() resolves positions to storage.

    Returns logits [1, T, vocab] if all_logits else [1, 1, vocab] for the final
    position. Slicing before lm_head skips a 151936-wide matmul on every prompt
    token during prefill, which is pure waste for greedy decoding.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "one request at a time"
    inner = model.model
    n_tokens = input_ids.shape[1]
    start = cache.length

    # Secure storage for this pass before any layer writes: all layers must
    # target the same slots, so growth cannot happen inside the loop.
    cache.reserve(n_tokens)

    hidden = inner.embed_tokens(input_ids)
    position_ids = torch.arange(
        start, start + n_tokens, device=hidden.device
    ).unsqueeze(0)
    cos, sin = inner.rotary_emb(hidden, position_ids)
    mask = causal_mask(n_tokens, start + n_tokens, hidden.dtype, hidden.device)

    for layer in inner.layers:
        attn = layer.self_attn
        if attn.sliding_window is not None:
            raise NotImplementedError(
                "sliding-window attention needs a windowed mask; this model uses it"
            )

        residual = hidden
        h = layer.input_layernorm(hidden)

        qkv_shape = (1, n_tokens, -1, attn.head_dim)
        q = attn.q_proj(h).view(qkv_shape).transpose(1, 2)
        k = attn.k_proj(h).view(qkv_shape).transpose(1, 2)
        v = attn.v_proj(h).view(qkv_shape).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # Our cache, our indexing. Every layer writes at the same `start`.
        k_all, v_all = cache.write(attn.layer_idx, k, v, start)

        attn_out = attend(attn, q, k_all, v_all, mask)
        hidden = residual + attn.o_proj(attn_out.reshape(1, n_tokens, -1))

        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    # Length advances once, after every layer has written at `start`.
    cache.commit(n_tokens)

    hidden = inner.norm(hidden)
    if not all_logits:
        hidden = hidden[:, -1:, :]
    return model.lm_head(hidden)
