"""Tests for dLux.abcd."""

import pytest

import dLux as dl

from .helpers import assert_differentiable, assert_jittable


@pytest.mark.parametrize(
    "element",
    [
        dl.ABCDFreeSpace(1.0),
        dl.ABCDLens(2.0),
        dl.ABCDMirror(3.0),
        dl.ABCDConjugatePlane(4.0),
    ],
)
def test_abcd_element_contract(element):
    matrix = assert_jittable(lambda value: value.abcd, element)
    assert matrix.shape == (2, 2)
    assert_differentiable(lambda value: value.abcd, element)
