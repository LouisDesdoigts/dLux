import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux.affine import (
    Affine,
    BaseAffineOperation,
    MatrixAffine,
    Rotation,
    Scaling,
    Shearing,
    Translation,
)


def test_matrix_affine_defaults_and_validation():
    identity = MatrixAffine()
    assert np.array_equal(identity.matrix, np.eye(2))
    assert np.array_equal(identity.offset, np.zeros(2))
    with pytest.raises(ValueError, match="matrix"):
        MatrixAffine(np.ones((3, 3)))
    with pytest.raises(ValueError, match="offset"):
        MatrixAffine(offset=np.ones(3))


def test_semantic_operations_and_validation():
    assert np.array_equal(Translation([1, 2]).offset, np.array([-1, -2]))
    assert Rotation(0.0).matrix.shape == (2, 2)
    assert np.array_equal(Scaling(2).matrix, np.eye(2) / 2)
    assert np.array_equal(Shearing([1, 2]).matrix, np.array([[1, 1], [2, 1]]))
    for constructor, value in [
        (Translation, [1]),
        (Rotation, [1]),
        (Shearing, [1]),
    ]:
        with pytest.raises(ValueError):
            constructor(value)
    with pytest.raises(ValueError, match="non-zero"):
        Scaling(0)


def test_affine_composition_and_application():
    coords = dlu.pixel_coords(4, 1.0)
    translation = Translation([0.1, -0.2])
    rotation = Rotation(0.2)
    composed = rotation @ translation
    assert isinstance(composed, Affine)
    assert np.allclose(composed(coords), rotation(translation(coords)))
    assert translation.__matmul__(object()) is NotImplemented


def test_affine_container_operations():
    translation = Translation([0.1, 0.2])
    affine = Affine({"shift": translation})
    assert affine.shift is translation
    assert np.array_equal(affine.translation, translation.translation)
    with pytest.raises(AttributeError):
        _ = affine.missing

    inserted = affine.insert(("turn", Rotation(0.1)), 1)
    assert list(inserted.operations) == ["shift", "turn"]
    assert list(inserted.remove("turn").operations) == ["shift"]
    assert isinstance(Affine(translation), Affine)
    assert isinstance(Affine.rotate(0.1), Affine)
    assert isinstance(Affine.translate([0, 0]), Affine)
    assert isinstance(Affine.scale(2), Affine)
    assert isinstance(Affine.shear([0, 0]), Affine)


def test_affine_matmul_and_gradients():
    left = Affine([Rotation(0.1)])
    right = Affine([Translation([0.1, 0.2])])
    assert isinstance(left @ right, Affine)
    assert left.__matmul__(object()) is NotImplemented
    coords = dlu.pixel_coords(4, 1.0)
    gradient = jax.grad(lambda angle: Affine.rotate(angle)(coords).sum())(0.1)
    assert np.isfinite(gradient)
    assert issubclass(Affine, BaseAffineOperation)
