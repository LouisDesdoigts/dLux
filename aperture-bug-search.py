import dLux
import jax.numpy as np
import jax.lax as jl
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt

from dLux.dev.constants import JWST_PRIMARY_SEGMENTS


def _grads_from_many_points(x1: float, y1: float) -> float:
    x_diffs = x1 - np.roll(x1, -1)
    y_diffs = y1 - np.roll(y1, -1)
    return y_diffs / x_diffs


def _offset(theta: float, threshold: float) -> float:
    comps = (theta < threshold).astype(float)
    return theta + comps * 2. * np.pi


def _is_orig_left_of_edge(ms: float, xs: float, ys: float) -> int:
    bc_orig = np.array([[0.]])
    dist_from_orig = _perp_dists_from_lines(ms, xs, ys, bc_orig, bc_orig)
    return np.sign(dist_from_orig)


def _perp_dists_from_lines(m: float, x1: float, y1: float, x: float, y: float) -> float:
    inf_case = (x - x1)
    gen_case = (m * inf_case - (y - y1)) / np.sqrt(1 + m ** 2)
    return np.where(np.isinf(m), inf_case, gen_case)


coordinates: float = dLux.utils.get_pixel_coordinates(128, 5. / 128)
vertices: float = JWST_PRIMARY_SEGMENTS[0][1]

vertices: float = vertices - np.mean(vertices, axis=0)

# +
bc_x1 = vertices[:, 0][:, None, None]
bc_y1 = vertices[:, 1][:, None, None]

bc_x = coordinates[0][None, :, :]
bc_y = coordinates[1][None, :, :]

# +
theta = np.arctan2(bc_y1, bc_x1)
offset_theta = _offset(theta, 0.)

sorted_inds = np.argsort(offset_theta.flatten())

sorted_x1 = bc_x1[sorted_inds]
sorted_y1 = bc_y1[sorted_inds]
sorted_m = _grads_from_many_points(sorted_x1, sorted_y1)
# -

# Correct until here.

dist_from_edges = _perp_dists_from_lines(sorted_m, sorted_x1, sorted_y1, bc_x, bc_y)

for dist in dist_from_edges:
    plt.imshow(dist > 0.)
    plt.colorbar()
    plt.show() 

dist_sgn = _is_orig_left_of_edge(sorted_m, sorted_x1, sorted_y1)
edges = (dist_from_edges * dist_sgn) > 0.

for edge in edges:
    plt.imshow(edge)
    plt.colorbar()
    plt.show()

plt.imshow(edges.prod(axis=0))


