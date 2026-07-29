"""Phase 1: a KV cache we allocate and own.

One contiguous buffer per request, sized up front for the worst case. This is
deliberately the naive layout, because its weaknesses are exactly what Phase 2's
paged allocator exists to fix:

  - you must reserve max_seq_len for every request before you know how long it
    will actually be, so a request that stops after 20 tokens still holds a
    2048-token reservation
  - two requests cannot share a buffer, so fragmentation is total
  - `utilization()` reports the waste, and is the number the paged version gets
    compared against later

The shape that matters: [num_layers, num_kv_heads, max_seq_len, head_dim].
num_kv_heads, not num_attention_heads. Qwen2.5-0.5B has 14 query heads sharing
2 KV heads, so getting this wrong costs 7x the memory and reads garbage.
"""

from __future__ import annotations

import torch

from .loader import model_shape_info


class ContiguousKVCache:
    """A single-request KV cache backed by one pre-allocated tensor per layer set."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (num_layers, num_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=dtype, device=self.device)
        self._length = 0

    @classmethod
    def for_model(cls, model, max_seq_len: int) -> "ContiguousKVCache":
        info = model_shape_info(model)
        param = next(model.parameters())
        return cls(
            num_layers=info["num_layers"],
            num_kv_heads=info["num_key_value_heads"],
            head_dim=info["head_dim"],
            max_seq_len=max_seq_len,
            dtype=param.dtype,
            device=param.device,
        )

    # --- state ---

    @property
    def length(self) -> int:
        """Number of token positions currently holding real data."""
        return self._length

    @property
    def capacity(self) -> int:
        return self.max_seq_len

    def free_slots(self) -> int:
        return self.max_seq_len - self._length

    def utilization(self) -> float:
        """Fraction of reserved positions holding real data. The headline waste number."""
        return self._length / self.max_seq_len if self.max_seq_len else 0.0

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def reset(self) -> None:
        """Reuse the buffer for a new request without reallocating.

        The stale numbers are left in place on purpose: nothing beyond
        `length` is ever read, and a test that only passes because the buffer was
        zeroed is a test that would miss a real indexing bug.
        """
        self._length = 0

    # --- writing / reading ---

    def reserve(self, n_tokens: int) -> None:
        """Ensure room for n_tokens more, called once before a forward pass.

        Trivial here because the whole buffer was reserved up front. The paged
        cache implements the same hook by allocating blocks, which is what lets
        one forward pass drive either cache unchanged.
        """
        if self._length + n_tokens > self.max_seq_len:
            raise KVCacheFull(
                f"need {self._length + n_tokens} positions, capacity {self.max_seq_len}"
            )

    def write(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store this layer's new K/V at [start, start+T) and return the full history.

        k, v arrive as [1, num_kv_heads, T, head_dim] straight out of the
        projections. The returned views cover [0, start+T) — everything the
        attention for this pass is allowed to look at.
        """
        assert k.shape == v.shape, f"k/v shape mismatch: {k.shape} vs {v.shape}"
        assert k.shape[0] == 1, "ContiguousKVCache holds one request at a time"
        n_tokens = k.shape[2]
        end = start + n_tokens
        if end > self.max_seq_len:
            raise KVCacheFull(
                f"need {end} positions but cache holds {self.max_seq_len}"
            )

        self.k[layer_idx, :, start:end, :] = k[0]
        self.v[layer_idx, :, start:end, :] = v[0]

        k_all = self.k[layer_idx, :, :end, :].unsqueeze(0)
        v_all = self.v[layer_idx, :, :end, :].unsqueeze(0)
        return k_all, v_all

    def commit(self, n_tokens: int) -> None:
        """Advance the length once per forward pass, after every layer has written.

        Separate from write() because all 24 layers write at the same `start`.
        Advancing inside write() would slide the offset out from under layer 1.
        """
        if self._length + n_tokens > self.max_seq_len:
            raise KVCacheFull(
                f"committing {n_tokens} would exceed capacity {self.max_seq_len}"
            )
        self._length += n_tokens


class KVCacheFull(RuntimeError):
    """Raised when a request needs more cache positions than are available."""
