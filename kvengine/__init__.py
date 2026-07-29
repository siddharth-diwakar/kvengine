"""kvengine - a from-scratch LLM serving engine, built up in phases."""

from .blocks import BlockManager, DoubleFree, OutOfBlocks
from .cache import ContiguousKVCache, KVCacheFull
from .decode import (
    AUTO_EOS,
    GenerationResult,
    greedy_no_cache,
    greedy_own_cache,
    greedy_paged,
    greedy_with_cache,
)
from .forward import forward_with_own_cache
from .loader import DEFAULT_MODEL, load_model, model_shape_info, pick_device
from .paged import DEFAULT_BLOCK_SIZE, PagedKVCache, PagedSequenceCache
from .reference import hf_greedy, strict_greedy_config

__all__ = [
    "AUTO_EOS",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_MODEL",
    "BlockManager",
    "ContiguousKVCache",
    "DoubleFree",
    "GenerationResult",
    "KVCacheFull",
    "OutOfBlocks",
    "PagedKVCache",
    "PagedSequenceCache",
    "forward_with_own_cache",
    "greedy_no_cache",
    "greedy_own_cache",
    "greedy_paged",
    "greedy_with_cache",
    "hf_greedy",
    "load_model",
    "model_shape_info",
    "pick_device",
    "strict_greedy_config",
]
