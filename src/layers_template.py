# TODO: Discuss the following abstract classes going forwards 
# with @LouisDesdoigts.
# NOTE: There is a lot of room to build a complex heirachy here 
# due to the generaility of the Layer concept.
# NOTE: So Optic -> Aperture actually makes sense from a 
# propagamitc perspective, just not a physical one.
# I guess that there is a drive to keep this simple because 
# users will be implementing there own Layers.
class Layer(eqx.Module):
    def __call__(self, parameters): # abstract

class CurvedOptic(Layer):
    def _curvature(self): # abstract

class Mirror(Layer):
    def _reflect(self, wavefront): # abstract
    def __call__(self, parameters): # concrete

class GaussianMirror(CurvedOptic, Mirror):
    def _curvature(self): # concrete
    def _reflect(self, wavefront): # concrete

class HyperbolicMirror(CurvedOptic, Mirror):
    def _curvature(self): # concrete
    def _reflect(self, wavefront): # concrete

class SphericalMirror(CurvedOptic, Mirror):
    def _curvature(self): # concrete
    def _reflect(self, wavefront): # concrete

class Lens(CurvedOptic):
    index_of_refraction : float 
    def _curvature(self): # abstract
    def _focus(self, wavefront): # concrete
    def __call__(self, parameters): # concrete

class GaussianLens(Lens):
    def _curvature(self): # concrete

class HyperbolicLens(Lens):
    def _curvature(self): # concrete 

class ParabolicLens(Lens):
    def _curvature(self): # concrete

class Aperture(Optic):
    def _area(self): # Still abtract 
    def __call__(self): # concrete

class CircularAperture(Aperture):
    radius : float
    def _area(self, wavefront):

class RegularPolygonalAperture(Aperture):
    number_of_sides : int
    radius : float
    def _area(self): # concrete

class ArbitraryAperture(Aperture):
    binary_mask : int
    def _area(self):
        return self.binary_mask

class PhasePlate(Optic):
    def _phase_shift(self): # abstract
    def __call__(self): # concrete

class ArbitraryPhasePlate(Optic):
    phase : array
    def _phase_shift(self):
        return self.phase

class ChromaticPhasePlate(Optic):
    def _phase_shift(self, wavefront): # concrete
        # dependence on wavelength 

