import pytest
import torch

from kvengine import DEFAULT_MODEL, load_model

# Tests run on CPU in float32 on purpose. The correctness anchor is an exact
# token match, and MPS/fp16 kernels reorder floating point work enough to flip
# near-tie argmax decisions. Speed is measured in the benchmark scripts, not here.
TEST_DEVICE = "cpu"
TEST_DTYPE = torch.float32


@pytest.fixture(scope="session")
def model_and_tokenizer():
    return load_model(DEFAULT_MODEL, device=TEST_DEVICE, dtype=TEST_DTYPE)


@pytest.fixture(scope="session")
def model(model_and_tokenizer):
    return model_and_tokenizer[0]


@pytest.fixture(scope="session")
def tokenizer(model_and_tokenizer):
    return model_and_tokenizer[1]


@pytest.fixture(scope="session")
def encode(tokenizer):
    def _encode(text: str) -> torch.Tensor:
        return tokenizer(text, return_tensors="pt").input_ids.to(TEST_DEVICE)

    return _encode
