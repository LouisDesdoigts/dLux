import dLux
import jax.numpy as np
import jax.lax as jl
import jax.random as jr

Array: type = np.ndarray


class IrregularPolygonalAperture(dLux.apertures.PolygonalAperture):
    vertices: Array
    
    def __init__(self   : dLux.apertures.dLux.apertures.ApertureLayer, 
            vertices    : Array,
            centre      : Array = np.array([0., 0.]), 
            strain      : Array = np.array([0., 0.]),
            compression : Array = np.array([1., 1.]),
            rotation    : Array = np.array(0.),
            occulting   : bool = False, 
            softening   : Array = np.array(1.),
            name        : str = "IrregularPolygonalAperture") -> dLux.apertures.ApertureLayer:
        super().__init__(
            centre = centre, 
            strain = strain, 
            compression = compression,
            rotation = rotation,
            occulting = occulting,
            softening = softening,
            name = name)
        self.vertices = np.array(vertices).astype(float)
        dLux.exceptions.validate_bc_attr_dims(
            (1, 2), self.vertices.shape, "vertices")
            
    def _grads_from_many_points(self: dLux.apertures.ApertureLayer, 
                                x1: float, 
                                y1: float) -> float:
        x_diffs = x1 - np.roll(x1, -1)
        y_diffs = y1 - np.roll(y1, -1)
        return y_diffs / x_diffs
    
    
    def _extent(self: dLux.apertures.ApertureLayer) -> float:
        verts = self.vertices
        dist_to_verts = np.hypot(verts[:, 1], verts[:, 0])
        return np.max(dist_to_verts)
    
    def _soft_edged(self: dLux.apertures.ApertureLayer, coordinates: float) -> float:
        # NOTE: see class docs.
        bc_x1 = self.vertices[:, 0][:, None, None]
        bc_y1 = self.vertices[:, 1][:, None, None]

        bc_x = coordinates[0][None, :, :]
        bc_y = coordinates[1][None, :, :]

        theta = np.arctan2(bc_y1, bc_x1)
        offset_theta = self._offset(theta, 0.)

        sorted_inds = np.argsort(offset_theta.flatten())

        sorted_x1 = bc_x1[sorted_inds]
        sorted_y1 = bc_y1[sorted_inds]
        sorted_m = self._grads_from_many_points(sorted_x1, sorted_y1)

        dist_from_edges = self._perp_dists_from_lines(sorted_m, sorted_x1, \
            sorted_y1, bc_x, bc_y)  
        dist_sgn = self._is_orig_left_of_edge(sorted_m, sorted_x1, sorted_y1)
        soft_edges = self._soften(dist_sgn * dist_from_edges)

        return (soft_edges).prod(axis=0)

    def _hard_edged(self: dLux.apertures.ApertureLayer, coordinates: Array) -> Array:
        # NOTE: see class docs.
        bc_x1 = self.vertices[:, 0][:, None, None]
        bc_y1 = self.vertices[:, 1][:, None, None]

        bc_x = coordinates[0][None, :, :]
        bc_y = coordinates[1][None, :, :]

        theta = np.arctan2(bc_y1, bc_x1)
        offset_theta = self._offset(theta, 0.)

        sorted_inds = np.argsort(offset_theta.flatten())

        sorted_x1 = bc_x1[sorted_inds]
        sorted_y1 = bc_y1[sorted_inds]
        sorted_m = self._grads_from_many_points(sorted_x1, sorted_y1)

        dist_from_edges = self._perp_dists_from_lines(sorted_m, sorted_x1, \
            sorted_y1, bc_x, bc_y)  
        dist_sgn = self._is_orig_left_of_edge(sorted_m, sorted_x1, sorted_y1)
        edges = (dist_from_edges * dist_sgn) > 0.

        return (edges).prod(axis=0)


