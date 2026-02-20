import os
import shutil
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="arrowdb_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def dim():
    """Default vector dimension for tests."""
    return 32


@pytest.fixture
def make_vector(dim):
    """Factory for creating test vectors."""
    import random

    def _make(seed=None):
        if seed is not None:
            random.seed(seed)
        return [random.gauss(0, 1) for _ in range(dim)]

    return _make
