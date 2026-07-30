"""Phase 2 + 6: the block allocator.

Pure bookkeeping — no tensors here. This file knows which fixed-size blocks are
free, how many sequences reference each one, and which blocks hold a cached token
prefix. The KV tensor itself lives in paged.py. Keeping them apart means the
allocator can be tested exhaustively without touching a model.

Phase 2: a slab pool. Fixed-size blocks, a free list, allocate/free under churn.
Phase 6: reference counting, so two requests sharing a prompt prefix can share the
blocks holding it, plus a content-addressed registry to find those blocks again.

Three invariants, all checked by check_invariants():

  1. a block is either free, reclaimable, or referenced — never two of those
  2. free + reclaimable + referenced == every block, with no duplicates
  3. a block is referenced if and only if its refcount is positive

The churn tests call it after every single operation.

Why sharing needs no copy-on-write
----------------------------------
Only *full* blocks are ever registered for sharing. A partially filled block still
has room, so a request will keep appending to it; sharing that would mean two
requests writing to the same storage. A full block can never be written again — the
next token goes to the next block — so shared blocks are immutable by construction
and no copy-on-write path is needed. vLLM needs CoW because it also shares partial
blocks when forking a sequence (beam search, n>1 sampling); with one linear
sequence per request, restricting sharing to full blocks removes that entire class
of bug for the cost of leaving up to block_size-1 tokens unshared per request.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from collections.abc import Sequence

# Digest that starts a prefix chain. Chaining means a block's identity depends on
# every token before it, so a block cached for positions 0-15 can never be
# mistaken for one covering positions 16-31. That matters because keys and values
# have their absolute position baked in by RoPE.
ROOT_DIGEST = b"\x00" * 16


def chain_digest(parent: bytes, token_ids: Sequence[int]) -> bytes:
    """Content address for a block: its tokens, chained onto its parent's digest."""
    h = hashlib.blake2b(parent, digest_size=16)
    h.update(struct.pack(f"<{len(token_ids)}i", *token_ids))
    return h.digest()


class OutOfBlocks(RuntimeError):
    """Not enough allocatable blocks to satisfy an allocation.

    Carries the numbers a scheduler needs in order to decide whether to wait,
    preempt another request, or reject this one outright.
    """

    def __init__(self, requested: int, available: int):
        super().__init__(f"requested {requested} blocks, only {available} available")
        self.requested = requested
        self.available = available


class DoubleFree(RuntimeError):
    """A block was released while not referenced by anything."""


class BlockManager:
    """Reference-counted slab pool over a fixed set of KV blocks.

    A block sits in exactly one of three states:

      free         nobody references it, it holds nothing worth keeping
      reclaimable  nobody references it, but it holds a cached prefix, so it is
                   kept until the space is actually needed (LRU order)
      referenced   one or more sequences are using it

    The reclaimable state is what makes prefix caching pay off across *sequential*
    requests and not just concurrent ones: a finished request's prefix blocks stay
    available for the next request that shares its prompt.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Reversed so the first pop() hands out block 0, which makes test
        # expectations readable.
        self._free: list[int] = list(reversed(range(num_blocks)))
        self._refcount: dict[int, int] = {}
        self._owner: dict[int, object] = {}  # creator, for debugging and free_owner

        # block -> None, in least-recently-released order.
        self._reclaimable: OrderedDict[int, None] = OrderedDict()
        # Content-addressed prefix registry, both directions.
        self._by_digest: dict[bytes, int] = {}
        self._digest_of: dict[int, bytes] = {}
        self._tokens_of: dict[int, tuple[int, ...]] = {}

        self.prefix_hits = 0
        self.prefix_misses = 0
        self.evictions = 0

    # --- state ---

    @property
    def num_free(self) -> int:
        """Blocks that can be handed out right now, including reclaimable ones."""
        return len(self._free) + len(self._reclaimable)

    @property
    def num_unused(self) -> int:
        """Blocks holding nothing at all."""
        return len(self._free)

    @property
    def num_reclaimable(self) -> int:
        """Blocks holding a cached prefix that nobody currently references."""
        return len(self._reclaimable)

    @property
    def num_allocated(self) -> int:
        return len(self._refcount)

    @property
    def total_slots(self) -> int:
        return self.num_blocks * self.block_size

    def pool_utilization(self) -> float:
        """Fraction of the pool currently referenced by some request."""
        return self.num_allocated / self.num_blocks if self.num_blocks else 0.0

    def owner_of(self, block_id: int) -> object | None:
        """The request that first allocated this block. Shared blocks keep the creator."""
        return self._owner.get(block_id)

    def refcount(self, block_id: int) -> int:
        return self._refcount.get(block_id, 0)

    def blocks_needed(self, n_tokens: int) -> int:
        """Blocks required to hold n_tokens, i.e. ceil division."""
        return (n_tokens + self.block_size - 1) // self.block_size

    # --- allocate / free ---

    def allocate(self, n_blocks: int, owner: object = None) -> list[int]:
        """Check out n_blocks, or raise without changing anything.

        Atomicity matters: a partial allocation that then raises would leak blocks
        on every rejected request, and a server rejects requests constantly under
        load. Capacity is checked before a single block moves.

        Truly-free blocks are spent before reclaimable ones, so a cached prefix is
        only discarded when the space is genuinely needed.
        """
        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative")
        if n_blocks > self.num_free:
            raise OutOfBlocks(n_blocks, self.num_free)

        taken: list[int] = []
        for _ in range(n_blocks):
            if self._free:
                taken.append(self._free.pop())
            else:
                taken.append(self._evict_lru())
        for b in taken:
            self._refcount[b] = 1
            self._owner[b] = owner
        return taken

    def incref(self, block_ids: Sequence[int]) -> None:
        """Take an additional reference on already-live or reclaimable blocks.

        This is prefix sharing: a second request adopts blocks the first one
        computed instead of computing them again.
        """
        ids = list(block_ids)
        for b in ids:
            if b not in self._refcount and b not in self._reclaimable:
                raise DoubleFree(f"block {b} is neither referenced nor reclaimable")
        for b in ids:
            if b in self._reclaimable:
                # Resurrect: it was cached and unreferenced, now it is in use again.
                del self._reclaimable[b]
                self._refcount[b] = 1
            else:
                self._refcount[b] += 1

    def free(self, block_ids) -> None:
        """Drop one reference per id. Blocks only return to the pool at zero.

        A block that reaches zero but holds a registered prefix becomes
        reclaimable rather than free, so the next request with the same prompt can
        still hit it.
        """
        ids = list(block_ids)
        for b in ids:
            if self._refcount.get(b, 0) <= 0:
                raise DoubleFree(f"block {b} is not referenced")
        for b in ids:
            self._refcount[b] -= 1
            if self._refcount[b] == 0:
                del self._refcount[b]
                if b in self._digest_of:
                    self._reclaimable[b] = None
                else:
                    self._owner.pop(b, None)
                    self._free.append(b)

    def free_owner(self, owner: object) -> list[int]:
        """Drop the references held by one request. Used when a request retires."""
        ids = [b for b, o in self._owner.items() if o == owner and b in self._refcount]
        self.free(ids)
        return ids

    def reset(self) -> None:
        self._free = list(reversed(range(self.num_blocks)))
        self._refcount.clear()
        self._owner.clear()
        self._reclaimable.clear()
        self._by_digest.clear()
        self._digest_of.clear()
        self._tokens_of.clear()

    # --- prefix registry ---

    def register_prefix(
        self, block_id: int, digest: bytes, token_ids: Sequence[int]
    ) -> bool:
        """Publish a full block so later requests can find it by content.

        Returns False if this prefix is already registered to another block; the
        caller's block simply stays private. Re-registering would orphan whichever
        entry lost, and two blocks holding identical keys and values is wasteful
        but harmless.
        """
        if len(token_ids) != self.block_size:
            raise ValueError(
                f"only full blocks are shareable: got {len(token_ids)} tokens, "
                f"block_size is {self.block_size}"
            )
        if digest in self._by_digest:
            return self._by_digest[digest] == block_id
        if self.refcount(block_id) <= 0:
            raise ValueError(f"cannot register unreferenced block {block_id}")

        self._by_digest[digest] = block_id
        self._digest_of[block_id] = digest
        self._tokens_of[block_id] = tuple(token_ids)
        return True

    def lookup_prefix(self, digest: bytes, token_ids: Sequence[int]) -> int | None:
        """Find a cached block by content, or None.

        The caller's tokens are verified against the stored ones. A blake2b
        collision here is astronomically unlikely, but serving the wrong keys and
        values would corrupt output silently rather than fail loudly, so it is
        worth the comparison.
        """
        block_id = self._by_digest.get(digest)
        if block_id is None:
            self.prefix_misses += 1
            return None
        if self._tokens_of.get(block_id) != tuple(token_ids):
            self.prefix_misses += 1
            return None
        self.prefix_hits += 1
        if block_id in self._reclaimable:
            self._reclaimable.move_to_end(block_id)
        return block_id

    def unregister_prefix(self, block_id: int) -> bool:
        """Withdraw a block from the shared registry.

        Needed when a block's contents stop being final — a speculative rollback
        that reaches back into a published block, for instance. After this, no new
        request can adopt it. An unreferenced block that loses its registration has
        nothing worth keeping, so it goes back to the free list.
        """
        digest = self._digest_of.pop(block_id, None)
        if digest is None:
            return False
        if self._by_digest.get(digest) == block_id:
            del self._by_digest[digest]
        self._tokens_of.pop(block_id, None)
        if block_id in self._reclaimable:
            del self._reclaimable[block_id]
            self._owner.pop(block_id, None)
            self._free.append(block_id)
        return True

    def _evict_lru(self) -> int:
        """Reclaim the least recently released cached block and unregister it."""
        block_id, _ = self._reclaimable.popitem(last=False)
        digest = self._digest_of.pop(block_id, None)
        if digest is not None:
            self._by_digest.pop(digest, None)
        self._tokens_of.pop(block_id, None)
        self._owner.pop(block_id, None)
        self.evictions += 1
        return block_id

    def prefix_hit_rate(self) -> float:
        total = self.prefix_hits + self.prefix_misses
        return self.prefix_hits / total if total else 0.0

    # --- invariants ---

    def check_invariants(self) -> None:
        """Assert the pool is internally consistent. Cheap enough to call often."""
        free_set = set(self._free)
        assert len(free_set) == len(self._free), "duplicate block in free list"

        reclaim_set = set(self._reclaimable)
        live_set = set(self._refcount)

        assert not (free_set & live_set), (
            f"blocks both free and referenced: {sorted(free_set & live_set)}"
        )
        assert not (free_set & reclaim_set), (
            f"blocks both free and reclaimable: {sorted(free_set & reclaim_set)}"
        )
        assert not (reclaim_set & live_set), (
            f"blocks both reclaimable and referenced: {sorted(reclaim_set & live_set)}"
        )
        assert free_set | reclaim_set | live_set == set(range(self.num_blocks)), (
            "blocks leaked: free + reclaimable + referenced does not cover the pool"
        )
        assert all(c > 0 for c in self._refcount.values()), (
            "a referenced block has a non-positive refcount"
        )
        # Every registry entry points at a real block, in both directions.
        for digest, block_id in self._by_digest.items():
            assert self._digest_of.get(block_id) == digest, (
                f"prefix registry disagrees with itself for block {block_id}"
            )
        # Reclaimable blocks are exactly the unreferenced registered ones.
        for block_id in reclaim_set:
            assert block_id in self._digest_of, (
                f"reclaimable block {block_id} holds no registered prefix"
            )

    def __repr__(self) -> str:
        return (
            f"BlockManager(num_blocks={self.num_blocks}, block_size={self.block_size}, "
            f"unused={self.num_unused}, reclaimable={self.num_reclaimable}, "
            f"referenced={self.num_allocated})"
        )
