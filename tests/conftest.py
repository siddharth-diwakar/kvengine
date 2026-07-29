import pytest
import torch

from kvengine import DEFAULT_MODEL, load_model

# Fixtures whose use means a test needs downloaded weights or a tokenizer.
_NEEDS_DOWNLOAD = {"model", "model_and_tokenizer", "tokenizer", "encode"}


def pytest_collection_modifyitems(items):
    """Tag every test `fast` or `slow` based on the fixtures it requests.

    Derived rather than hand-annotated on purpose: a new test gets classified
    correctly without anyone remembering to mark it, and a test can never drift
    into the fast suite and start silently downloading a model there.
    """
    for item in items:
        needs_model = _NEEDS_DOWNLOAD & set(getattr(item, "fixturenames", ()))
        item.add_marker(pytest.mark.slow if needs_model else pytest.mark.fast)

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
