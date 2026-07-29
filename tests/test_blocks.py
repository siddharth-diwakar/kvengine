"""Phase 2: the block allocator, tested without a model.

No tensors and no weights here, so these are fast enough to hammer. The point is
to prove the pool never corrupts itself under churn — the exact class of bug that
would otherwise surface as mysteriously wrong generated text.
"""

import random

import pytest

from kvengine import BlockManager, DoubleFree, OutOfBlocks


def test_starts_fully_free():
    m = BlockManager(num_blocks=8, block_size=16)
    assert m.num_free == 8
    assert m.num_allocated == 0
    assert m.total_slots == 128
    m.check_invariants()


def test_allocate_and_free_roundtrip():
    m = BlockManager(4, 16)
    blocks = m.allocate(3, owner="a")
    assert len(set(blocks)) == 3
    assert m.num_free == 1
    m.check_invariants()

    m.free(blocks)
    assert m.num_free == 4
    assert m.num_allocated == 0
    m.check_invariants()


def test_blocks_needed_rounds_up():
    m = BlockManager(100, 16)
    assert m.blocks_needed(0) == 0
    assert m.blocks_needed(1) == 1
    assert m.blocks_needed(16) == 1
    assert m.blocks_needed(17) == 2
    assert m.blocks_needed(32) == 2


def test_a_block_is_never_owned_twice():
    """The core slab-pool invariant."""
    m = BlockManager(16, 8)
    a = m.allocate(6, owner="a")
    b = m.allocate(6, owner="b")
    assert not set(a) & set(b), "same block handed to two requests"
    for blk in a:
        assert m.owner_of(blk) == "a"
    for blk in b:
        assert m.owner_of(blk) == "b"
    m.check_invariants()


def test_oom_raises_and_changes_nothing():
    """A rejected allocation must not leak blocks.

    Servers reject constantly under load. If a failed allocate() popped some
    blocks before discovering it could not finish, the pool would bleed capacity
    on every rejection until it wedged.
    """
    m = BlockManager(4, 16)
    held = m.allocate(3, owner="a")
    before_free = m.num_free

    with pytest.raises(OutOfBlocks) as exc:
        m.allocate(2, owner="b")

    assert exc.value.requested == 2
    assert exc.value.available == 1
    assert m.num_free == before_free, "failed allocation leaked blocks"
    assert m.num_allocated == len(held)
    m.check_invariants()

    # The pool is still usable for a request that does fit.
    assert len(m.allocate(1, owner="c")) == 1
    m.check_invariants()


def test_exact_fit_allocation_succeeds():
    m = BlockManager(4, 16)
    assert len(m.allocate(4, owner="a")) == 4
    assert m.num_free == 0
    with pytest.raises(OutOfBlocks):
        m.allocate(1)
    m.check_invariants()


def test_zero_block_allocation_is_a_noop():
    """A request whose prompt is empty must not be a special case upstream."""
    m = BlockManager(4, 16)
    assert m.allocate(0, owner="a") == []
    assert m.num_free == 4
    m.check_invariants()


def test_double_free_is_rejected():
    m = BlockManager(4, 16)
    blocks = m.allocate(2, owner="a")
    m.free(blocks)
    with pytest.raises(DoubleFree):
        m.free(blocks)
    m.check_invariants()


def test_free_is_atomic_across_the_batch():
    """Freeing a list containing one bad id must free none of them.

    Otherwise a buggy caller leaves the pool half-updated, which is far harder to
    debug than an outright rejection.
    """
    m = BlockManager(8, 16)
    good = m.allocate(3, owner="a")
    with pytest.raises(DoubleFree):
        m.free(good + [7])  # block 7 is free, not allocated
    assert m.num_allocated == 3, "partial free left the pool inconsistent"
    m.check_invariants()


def test_free_owner_releases_only_that_request():
    m = BlockManager(16, 8)
    m.allocate(4, owner="a")
    b = m.allocate(4, owner="b")
    released = m.free_owner("a")

    assert len(released) == 4
    assert m.num_allocated == 4
    assert all(m.owner_of(blk) == "b" for blk in b)
    m.check_invariants()


def test_lifo_reuse_returns_the_hottest_block():
    m = BlockManager(8, 16)
    first = m.allocate(1, owner="a")
    m.free(first)
    again = m.allocate(1, owner="b")
    assert again == first, "expected the most recently freed block back"


def test_reset_returns_everything():
    m = BlockManager(8, 16)
    m.allocate(5, owner="a")
    m.reset()
    assert m.num_free == 8
    assert m.num_allocated == 0
    m.check_invariants()


def test_churn_preserves_invariants():
    """Randomised allocate/free churn, invariants checked after every operation.

    This is the test that would have caught every pool bug I have written before.
    Fixed seed so a failure is reproducible.
    """
    rng = random.Random(1234)
    m = BlockManager(num_blocks=64, block_size=8)
    live: dict[str, list[int]] = {}
    next_id = 0
    allocations = 0
    rejections = 0

    for _ in range(4000):
        if live and rng.random() < 0.45:
            owner = rng.choice(list(live))
            m.free(live.pop(owner))
        else:
            owner = f"r{next_id}"
            next_id += 1
            want = rng.randint(1, 12)
            try:
                live[owner] = m.allocate(want, owner=owner)
                allocations += 1
            except OutOfBlocks:
                rejections += 1
        m.check_invariants()

        # No block appears in two live block tables, ever.
        seen: set[int] = set()
        for blocks in live.values():
            assert not seen & set(blocks), "block double-owned during churn"
            seen.update(blocks)

    assert allocations > 100, "churn did not actually allocate much"
    assert rejections > 0, "churn never hit the pool limit, so OOM went untested"

    for blocks in live.values():
        m.free(blocks)
    assert m.num_free == 64, "blocks leaked over the run"
    m.check_invariants()


def test_churn_never_loses_capacity():
    """Fully draining and releasing the pool repeatedly must return to full free."""
    m = BlockManager(32, 4)
    for _ in range(50):
        held = m.allocate(32, owner="hog")
        assert m.num_free == 0
        m.free(held)
        assert m.num_free == 32
    m.check_invariants()


def test_rejects_nonsense_construction():
    with pytest.raises(ValueError):
        BlockManager(0, 16)
    with pytest.raises(ValueError):
        BlockManager(16, 0)
