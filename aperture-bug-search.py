import dLux
import jax.numpy as np
import jax.lax as jl
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt

from dLux.dev.constants import JWST_PRIMARY_SEGMENTS

coordinates: float = dLux.utils.get_pixel_positions(128, 5. / 128)
vertices: float = JWST_PRIMARY_SEGMENTS[0][1]

# +
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

for edge in edges:
    plt.imshow(edge)
    plt.show()

return (edges).prod(axis=0)
