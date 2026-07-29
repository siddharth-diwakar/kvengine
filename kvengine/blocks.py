"""Phase 2: the block allocator.

Pure bookkeeping — no tensors here. This file knows which fixed-size blocks are
free and which request owns each one; the KV tensor itself lives in paged.py.
Keeping them apart means the allocator can be tested exhaustively without
touching a model.

This is a slab pool. Fixed-size blocks, a free list, allocate/free under churn.
The two invariants that matter are the ones a slab pool always has:

  1. a block is owned by at most one request at a time
  2. free + allocated == every block, always, with no duplicates

Both are checked by check_invariants(), which the churn tests call after every
single operation.
"""

from __future__ import annotations


class OutOfBlocks(RuntimeError):
    """Not enough free blocks to satisfy an allocation.

    Carries the numbers a scheduler needs in order to decide whether to wait,
    preempt another request, or reject this one outright.
    """

    def __init__(self, requested: int, available: int):
        super().__init__(f"requested {requested} blocks, only {available} free")
        self.requested = requested
        self.available = available


class DoubleFree(RuntimeError):
    """A block was returned to the pool while not allocated."""


class BlockManager:
    """Free list over a fixed pool of KV blocks.

    LIFO on purpose: the most recently freed block is the most likely to still
    be warm in cache, and reusing it immediately makes use-after-free bugs show
    up loudly in tests rather than lurking behind stale-but-plausible data.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Reversed so the first pop() hands out block 0, which makes test
        # expectations readable.
        self._free: list[int] = list(reversed(range(num_blocks)))
        self._owner: dict[int, object] = {}

    # --- state ---

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return len(self._owner)

    @property
    def total_slots(self) -> int:
        return self.num_blocks * self.block_size

    def pool_utilization(self) -> float:
        """Fraction of the pool currently checked out by some request."""
        return self.num_allocated / self.num_blocks if self.num_blocks else 0.0

    def owner_of(self, block_id: int) -> object | None:
        return self._owner.get(block_id)

    def blocks_needed(self, n_tokens: int) -> int:
        """Blocks required to hold n_tokens, i.e. ceil division."""
        return (n_tokens + self.block_size - 1) // self.block_size

    # --- allocate / free ---

    def allocate(self, n_blocks: int, owner: object = None) -> list[int]:
        """Check out n_blocks, or raise without changing anything.

        Atomicity matters: a partial allocation that then raises would leak
        blocks on every rejected request, and a server rejects requests
        constantly under load. Capacity is checked before a single pop.
        """
        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative")
        if n_blocks > len(self._free):
            raise OutOfBlocks(n_blocks, len(self._free))
        taken = [self._free.pop() for _ in range(n_blocks)]
        for b in taken:
            self._owner[b] = owner
        return taken

    def free(self, block_ids) -> None:
        """Return blocks to the pool. Rejects double-frees loudly."""
        ids = list(block_ids)
        for b in ids:
            if b not in self._owner:
                raise DoubleFree(f"block {b} is not allocated")
        for b in ids:
            del self._owner[b]
            self._free.append(b)

    def free_owner(self, owner: object) -> list[int]:
        """Release everything held by one request. Used when a request retires."""
        ids = [b for b, o in self._owner.items() if o == owner]
        self.free(ids)
        return ids

    def reset(self) -> None:
        self._free = list(reversed(range(self.num_blocks)))
        self._owner.clear()

    # --- invariants ---

    def check_invariants(self) -> None:
        """Assert the pool is internally consistent. Cheap enough to call often."""
        free_set = set(self._free)
        assert len(free_set) == len(self._free), "duplicate block in free list"

        owned = set(self._owner)
        assert not (free_set & owned), (
            f"blocks both free and allocated: {sorted(free_set & owned)}"
        )
        assert free_set | owned == set(range(self.num_blocks)), (
            "blocks leaked: free + allocated does not cover the pool"
        )

    def __repr__(self) -> str:
        return (
            f"BlockManager(num_blocks={self.num_blocks}, block_size={self.block_size}, "
            f"free={self.num_free}, allocated={self.num_allocated})"
        )
