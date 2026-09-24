"""Keep upstream NaN checking local to released tests."""

import jax
import pytest


@pytest.fixture(autouse=True)
def released_nan_checks():
    with jax.debug_nans(True):
        yield
