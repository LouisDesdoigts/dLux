import jax.numpy as np
import pytest
from dLux.layers import CoordTransform


def test_coord_transform():
    CoordTransform().calculate(1, 16)
    CoordTransform([0.0, 0.0], np.pi, [1, 1], [1, 1]).calculate(1, 16)
    with pytest.raises(ValueError):
        CoordTransform(translation=[0.0])
    with pytest.raises(ValueError):
        CoordTransform(rotation=[0.0])
    with pytest.raises(ValueError):
        CoordTransform(compression=[0.0])
    with pytest.raises(ValueError):
        CoordTransform(shear=[0.0])
