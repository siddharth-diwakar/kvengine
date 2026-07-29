"""Phase 2: the paged KV cache.

One big tensor of N fixed-size blocks, allocated at startup and handed out a
block at a time. A request's history is no longer contiguous — it is a list of
block ids (its *block table*), and attention has to gather K/V from wherever
those blocks happen to live.

Tensor layout:

    [num_layers, num_blocks, block_size, num_kv_heads, head_dim]

Blocks and slots-within-block are adjacent dims, so merging them gives a flat
slot index space (`block_id * block_size + offset`) that scatter writes can
address directly. That is the same reason vLLM lays its cache out this way.

PagedSequenceCache exposes exactly the interface ContiguousKVCache does
(length / reserve / write / commit), so the forward pass from phase 1 drives
either one without modification. The whole difference between "contiguous" and
"paged" is hidden behind write().

Known difference from real vLLM: gathering materializes the full K/V history for
a layer before attention runs. vLLM instead ships a fused CUDA kernel that
attends block-by-block and never materializes anything. Ours is the honest
portable version — correct, and slower by a memory-traffic constant. See the
README for the tradeoff.
"""

from __future__ import annotations

import torch

from .blocks import BlockManager, OutOfBlocks
from .loader import model_shape_info

DEFAULT_BLOCK_SIZE = 16


class PagedKVCache:
    """The shared pool: one tensor of blocks plus the allocator that owns them."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=dtype, device=self.device)

        # Flat views over (block, slot) for scatter writes. Views, not copies:
        # writing through these mutates self.k / self.v.
        flat = (num_layers, num_blocks * block_size, num_kv_heads, head_dim)
        self.k_flat = self.k.view(flat)
        self.v_flat = self.v.view(flat)

        self.manager = BlockManager(num_blocks, block_size)

    @classmethod
    def for_model(
        cls,
        model,
        num_blocks: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> "PagedKVCache":
        info = model_shape_info(model)
        param = next(model.parameters())
        return cls(
            num_layers=info["num_layers"],
            num_kv_heads=info["num_key_value_heads"],
            head_dim=info["head_dim"],
            num_blocks=num_blocks,
            block_size=block_size,
            dtype=param.dtype,
            device=param.device,
        )

    # --- capacity accounting ---

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def bytes_per_block(self) -> int:
        return self.nbytes() // self.num_blocks

    @property
    def total_slots(self) -> int:
        return self.num_blocks * self.block_size

    def new_sequence(self, request_id: object) -> "PagedSequenceCache":
        return PagedSequenceCache(self, request_id)

    def stats(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "blocks_allocated": self.manager.num_allocated,
            "blocks_free": self.manager.num_free,
            "pool_utilization": self.manager.pool_utilization(),
            "mib": self.nbytes() / 2**20,
        }

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(blocks={self.num_blocks}x{self.block_size}, "
            f"free={self.manager.num_free}, {self.nbytes() / 2**20:.1f} MiB)"
        )


class PagedSequenceCache:
    """One request's view of the pool: a block table plus a token count.

    Drop-in replacement for ContiguousKVCache from the forward pass's point of
    view, which is the payoff of having defined that interface in phase 1.
    """

    def __init__(self, pool: PagedKVCache, request_id: object):
        self.pool = pool
        self.request_id = request_id
        self.blocks: list[int] = []
        self._length = 0
        # Block table as a device tensor, so the gather does not re-upload a
        # Python list on every layer of every decode step.
        self._block_ids = torch.empty(0, dtype=torch.long, device=pool.device)
        self._pending_slots: torch.Tensor | None = None
        self._pending_n = 0

    # --- state ---

    @property
    def length(self) -> int:
        return self._length

    @property
    def block_size(self) -> int:
        return self.pool.block_size

    @property
    def capacity(self) -> int:
        """Slots this request currently holds, including the unused tail."""
        return len(self.blocks) * self.block_size

    def free_slots(self) -> int:
        return self.capacity - self._length

    def utilization(self) -> float:
        """Real tokens / slots held. Only internal fragmentation in the last block.

        Contrast with ContiguousKVCache.utilization(), where the denominator is a
        worst-case reservation. Here it is bounded by block_size - 1 wasted slots
        per request no matter how long the request turns out to be.
        """
        return self._length / self.capacity if self.blocks else 0.0

    def nbytes(self) -> int:
        return len(self.blocks) * self.pool.bytes_per_block()

    # --- allocation ---

    def reserve(self, n_tokens: int) -> None:
        """Grow the block table to cover length + n_tokens, then plan the write.

        Called once per forward pass, before any layer writes. Growing here
        rather than inside write() is essential: write() runs 24 times per pass
        and they must all target the same slots.
        """
        needed_end = self._length + n_tokens
        needed_blocks = self.pool.manager.blocks_needed(needed_end)
        extra = needed_blocks - len(self.blocks)
        if extra > 0:
            # Raises OutOfBlocks without mutating anything if the pool is full.
            new_blocks = self.pool.manager.allocate(extra, owner=self.request_id)
            self.blocks.extend(new_blocks)
            self._block_ids = torch.tensor(
                self.blocks, dtype=torch.long, device=self.pool.device
            )
        self._pending_slots = self.slot_indices(self._length, n_tokens)
        self._pending_n = n_tokens

    def slot_indices(self, start: int, n_tokens: int) -> torch.Tensor:
        """Flat slot index for each logical position in [start, start+n).

        This translation is the whole idea of paging: logical position -> block
        table lookup -> physical slot. Everything else is plumbing.
        """
        pos = torch.arange(start, start + n_tokens, device=self.pool.device)
        block_slot = self._block_ids[pos // self.block_size]
        return block_slot * self.block_size + (pos % self.block_size)

    # --- write / gather ---

    def write(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter this layer's new K/V into its blocks, then gather the history.

        k, v arrive as [1, num_kv_heads, T, head_dim]. Returns the full history
        for this layer as [1, num_kv_heads, start+T, head_dim] — contiguous
        again, even though it is stored in scattered blocks.
        """
        assert k.shape == v.shape, f"k/v shape mismatch: {k.shape} vs {v.shape}"
        assert k.shape[0] == 1, "one request per sequence cache"
        n_tokens = k.shape[2]
        assert start == self._length, (
            f"write at {start} but sequence length is {self._length}"
        )
        assert self._pending_slots is not None and self._pending_n == n_tokens, (
            "reserve() must be called with this pass's token count before write()"
        )

        # [1, H, T, D] -> [T, H, D] to match the flat slot layout.
        k_src = k[0].transpose(0, 1).contiguous()
        v_src = v[0].transpose(0, 1).contiguous()
        self.pool.k_flat[layer_idx].index_copy_(0, self._pending_slots, k_src)
        self.pool.v_flat[layer_idx].index_copy_(0, self._pending_slots, v_src)

        return self._gather(layer_idx, start + n_tokens)

    def _gather(self, layer_idx: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect [0, end) for this layer out of the request's blocks.

        Reads whole blocks and trims the tail, rather than gathering slot by
        slot: one strided read of block_size rows beats block_size scattered
        reads, and the trim is free.
        """
        n_blocks = self.pool.manager.blocks_needed(end)
        ids = self._block_ids[:n_blocks]

        k_blocks = self.pool.k[layer_idx][ids]  # [nb, block_size, H, D]
        v_blocks = self.pool.v[layer_idx][ids]
        shape = (n_blocks * self.block_size, self.pool.num_kv_heads, self.pool.head_dim)

        k_all = k_blocks.reshape(shape)[:end].transpose(0, 1).unsqueeze(0)
        v_all = v_blocks.reshape(shape)[:end].transpose(0, 1).unsqueeze(0)
        return k_all.contiguous(), v_all.contiguous()

    def commit(self, n_tokens: int) -> None:
        """Advance the length once, after every layer has written."""
        if self._length + n_tokens > self.capacity:
            raise OutOfBlocks(
                self.pool.manager.blocks_needed(self._length + n_tokens) - len(self.blocks),
                self.pool.manager.num_free,
            )
        self._length += n_tokens
        self._pending_slots = None
        self._pending_n = 0

    def truncate(self, n_tokens: int) -> None:
        """Roll back to n_tokens and return any blocks that are now entirely unused.

        Where paging pays off a second time: rejecting drafted tokens is just
        handing blocks back to the pool. A contiguous cache would have to keep the
        whole reservation regardless.
        """
        if not 0 <= n_tokens <= self._length:
            raise ValueError(f"cannot truncate to {n_tokens}, length is {self._length}")

        keep = self.pool.manager.blocks_needed(n_tokens)
        if keep < len(self.blocks):
            self.pool.manager.free(self.blocks[keep:])
            self.blocks = self.blocks[:keep]
            self._block_ids = torch.tensor(
                self.blocks, dtype=torch.long, device=self.pool.device
            )
        self._length = n_tokens
        self._pending_slots = None
        self._pending_n = 0

    # --- teardown ---

    def free(self) -> None:
        """Return every block to the pool. Idempotent, so double-retire is safe."""
        if self.blocks:
            self.pool.manager.free(self.blocks)
            self.blocks = []
            self._block_ids = torch.empty(0, dtype=torch.long, device=self.pool.device)
        self._length = 0
        self._pending_slots = None
        self._pending_n = 0

    def __repr__(self) -> str:
        return (
            f"PagedSequenceCache(req={self.request_id!r}, len={self._length}, "
            f"blocks={len(self.blocks)}, util={self.utilization():.1%})"
        )
