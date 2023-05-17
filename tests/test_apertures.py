
import jax.numpy as np
from jax import Array, config
import pytest
import dLux
from inspect import signature
import dLux.utils as dlu

config.update("jax_debug_nans", True)


def _test_transmission(constructor):
    """Tests the tranmission function of some input aperture"""
    constructor().transmission(16, 1)

def _test_base_dynamic_aperture_constructor(constructor):
    constructor()
    with pytest.raises(ValueError):
        constructor(centre=[0])
    with pytest.raises(ValueError):
        constructor(shear=[0])
    with pytest.raises(ValueError):
        constructor(compression=[0])
    
    # Have to add this here since CircularAperture is a special case
    if 'rotation' in signature(constructor).parameters.values():
        with pytest.raises(ValueError):
            constructor(rotation=[0])

def _test_call(constructor, wf_constructor):
    """Tests the __call__ function of some input aperture"""
    constructor()(wf_constructor())
    if 'normalise' in signature(constructor).parameters.values():
        constructor(normalise=False)(wf_constructor())

def _test_dyanmic_aperture_constructor(constructor):
    """Tests the constructor of some input aperture"""
    constructor()
    with pytest.raises(ValueError):
        constructor(softening=[0, 0])

def _test_make_static(constructor):
    """Tests the make_static function of some input aperture"""
    constructor().make_static(16, 1)

def _test_basis(constructor):
    """Tests the basis function of some input aperture"""
    constructor()._basis(dlu.pixel_coords(16, 1/16))

def _test_opd(constructor):
    """Tests the opd function of some input aperture"""
    constructor()._opd(dlu.pixel_coords(16, 1/16))


class TestCircularAperture():
    """Tests the CircularAperture class."""

    def test_constructor(self, create_circular_aperture):
        """Tests the constructor."""
        _test_base_dynamic_aperture_constructor(create_circular_aperture)
        _test_dyanmic_aperture_constructor(create_circular_aperture)

    def test_call(self, create_circular_aperture, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_circular_aperture, create_wavefront)
    
    def test_transmission(self, create_circular_aperture):
        """Tests the transmission method."""
        _test_transmission(create_circular_aperture)
    
    def test_make_static(self, create_circular_aperture):
        """Tests the make_static method."""
        _test_make_static(create_circular_aperture)


class TestRectangularAperture():
    """Tests the RectangularAperture class."""

    def test_constructor(self, create_rectangular_aperture):
        """Tests the constructor."""
        _test_base_dynamic_aperture_constructor(create_rectangular_aperture)
        _test_dyanmic_aperture_constructor(create_rectangular_aperture)

    def test_call(self, create_rectangular_aperture, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_rectangular_aperture, create_wavefront)
    
    def test_transmission(self, create_rectangular_aperture):
        """Tests the transmission method."""
        _test_transmission(create_rectangular_aperture)
    
    def test_make_static(self, create_rectangular_aperture):
        """Tests the make_static method."""
        _test_make_static(create_rectangular_aperture)


class TestRegPolyAperture():
    """Tests the RegPolyAperture class."""

    def test_constructor(self, create_reg_poly_aperture):
        """Tests the constructor."""
        _test_base_dynamic_aperture_constructor(create_reg_poly_aperture)
        _test_dyanmic_aperture_constructor(create_reg_poly_aperture)
    
    def test_call(self, create_reg_poly_aperture, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_reg_poly_aperture, create_wavefront)
    
    def test_transmission(self, create_reg_poly_aperture):
        """Tests the transmission method."""
        _test_transmission(create_reg_poly_aperture)
    
    def test_make_static(self, create_reg_poly_aperture):
        """Tests the make_static method."""
        _test_make_static(create_reg_poly_aperture)


class TestIrregPolyAperture():
    """Tests the IrregPolyAperture class."""

    def test_constructor(self, create_irreg_poly_aperture):
        """Tests the constructor."""
        _test_base_dynamic_aperture_constructor(create_irreg_poly_aperture)
        _test_dyanmic_aperture_constructor(create_irreg_poly_aperture)
    
    def test_call(self, create_irreg_poly_aperture, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_irreg_poly_aperture, create_wavefront)
    
    def test_transmission(self, create_irreg_poly_aperture):
        """Tests the transmission method."""
        _test_transmission(create_irreg_poly_aperture)

    def test_make_static(self, create_irreg_poly_aperture):
        """Tests the make_static method."""
        _test_make_static(create_irreg_poly_aperture)


class TestAberratedAperture():
    """Tests the AberratedAperture class."""

    def test_constructor(self, create_aberrated_aperture):
        """Tests the constructor."""
        create_aberrated_aperture()
    
    def test_call(self, create_aberrated_aperture, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_aberrated_aperture, create_wavefront)
    
    def test_transmission(self, create_aberrated_aperture):
        """Tests the transmission method."""
        _test_transmission(create_aberrated_aperture)
    
    def test_make_static(self, create_aberrated_aperture):
        """Tests the make_static method."""
        _test_make_static(create_aberrated_aperture)
    
    def test_basis(self, create_aberrated_aperture):
        """Tests the basis method."""
        _test_basis(create_aberrated_aperture)
    
    def test_opd(self, create_aberrated_aperture):
        """Tests the _opd method."""
        _test_opd(create_aberrated_aperture)
        
    
class TestUniformSpider():
    """Tests the UniformSpider class."""

    def test_constructor(self, create_uniform_spider):
        """Tests the constructor."""
        create_uniform_spider()
    
    def test_call(self, create_uniform_spider, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_uniform_spider, create_wavefront)
    
    def test_transmission(self, create_uniform_spider):
        """Tests the transmission method."""
        _test_transmission(create_uniform_spider)
    
    def test_make_static(self, create_uniform_spider):
        """Tests the make_static method."""
        _test_make_static(create_uniform_spider)


class TestCompoundAperture():
    """Tests the CompoundAperture class."""

    def test_constructor(self, create_compound_aperture):
        """Tests the constructor."""
        create_compound_aperture()
    
    def test_call(self, create_compound_aperture, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_compound_aperture, create_wavefront)
    
    def test_transmission(self, create_compound_aperture):
        """Tests the transmission method."""
        _test_transmission(create_compound_aperture)
    
    def test_make_static(self, create_compound_aperture):
        """Tests the make_static method."""
        _test_make_static(create_compound_aperture)

    def test_basis(self, create_compound_aperture):
        """Tests the basis method."""
        _test_basis(create_compound_aperture)
    
    def test_opd(self, create_compound_aperture):
        """Tests the _opd method."""
        _test_opd(create_compound_aperture)


class TestMultiAperture():
    """Tests the MultiAperture class."""

    def test_constructor(self, create_multi_aperture):
        """Tests the constructor."""
        create_multi_aperture()
    
    def test_call(self, create_multi_aperture, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_multi_aperture, create_wavefront)
    
    def test_transmission(self, create_multi_aperture):
        """Tests the transmission method."""
        _test_transmission(create_multi_aperture)
    
    def test_make_static(self, create_multi_aperture):
        """Tests the make_static method."""
        _test_make_static(create_multi_aperture)

    def test_basis(self, create_multi_aperture):
        """Tests the basis method."""
        _test_basis(create_multi_aperture)
    
    def test_opd(self, create_multi_aperture):
        """Tests the _opd method."""
        _test_opd(create_multi_aperture)