"""Model and tokenizer loading, plus device/dtype selection.

Kept deliberately small: every phase of the project loads weights the same way,
so the only thing that changes between phases is how the KV cache is managed.
"""

from __future__ import annotations

import os

# HuggingFace serves weight files through Xet storage by default, and the
# hf_xet client hangs indefinitely on this machine (metadata downloads fine;
# plain HTTPS to the same CDN pulls at normal speed). Forcing the ordinary HTTP
# download path avoids a silent multi-minute stall on first load. Set
# HF_HUB_DISABLE_XET=0 in the environment to opt back in.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def pick_device(prefer: str = "auto") -> torch.device:
    """Resolve a device string. 'auto' prefers MPS on Apple silicon, else CPU."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    model_id: str = DEFAULT_MODEL,
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
):
    """Load a causal LM in eval mode along with its tokenizer.

    Defaults to float32 because the correctness anchor for this project is an
    exact token-for-token match against HuggingFace generation. Reduced
    precision makes near-tie argmax decisions flip between runs, which turns a
    real bug and a rounding artifact into the same symptom.
    """
    dev = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(dev)
    model.eval()
    return model, tokenizer


def model_shape_info(model) -> dict:
    """The handful of config numbers the KV cache layout depends on.

    Qwen2.5 uses grouped-query attention, so the cache is sized by
    num_key_value_heads (4), not num_attention_heads (14). Getting this wrong is
    the classic first bug in a hand-rolled cache, so it is surfaced explicitly.
    """
    cfg = model.config
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    return {
        "num_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }
