from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux import CoordTransform


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
