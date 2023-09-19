import pytest
import jax.numpy as np
from jax import random
from dLux.utils.math import factorial, triangular_number, eval_basis, nandiv


@pytest.fixture
def n():
    return 5.0


@pytest.fixture
def m():
    return 3


@pytest.fixture
def basis():
    key = random.PRNGKey(0)
    return random.normal(key, (5, 10, 10))


@pytest.fixture
def coefficients():
    key = random.PRNGKey(1)
    return random.normal(key, (5,))


@pytest.fixture
def a():
    key = random.PRNGKey(2)
    return random.normal(key, (10, 10))


@pytest.fixture
def b():
    key = random.PRNGKey(3)
    return random.normal(key, (10, 10))


@pytest.fixture
def fill():
    return np.inf


def test_factorial(n):
    result = factorial(n)
    assert np.allclose(result, 120.0)


def test_triangular_number(m):
    result = triangular_number(m)
    assert result == m * (m + 1) / 2


def test_eval_basis(basis, coefficients):
    result = eval_basis(basis, coefficients)
    assert result.shape == (10, 10)


def test_nandiv(a, b, fill):
    result = nandiv(a, b, fill)
    assert result.shape == (10, 10)
