import jax
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "inferno"

n: int = 7
rmax: float = 1.
alpha: float = np.pi / n # Half the angular disp of one wedge

npix: int = 100
grid: float = np.linspace(0, 2., npix) - 1.
coords: float = np.meshgrid(grid, grid)

neg_pi_to_pi_phi: float = np.arctan2(coords[1], coords[0]) 
phi: float = neg_pi_to_pi_phi + 2. * (neg_pi_to_pi_phi < 0.) * np.pi
rho: float = np.hypot(coords[0], coords[1])

i: int = np.arange(n)
low_bound: float = 2. * i * alpha
top_bound: float = 2. * (i + 1.) * alpha  

wedge: float = ((low_bound[:, None, None] < phi) & (phi <= top_bound[:, None, None])).astype(float)
min_inv_m: float = np.tan((2. * i + 1.) * alpha)
x_proj: float = np.cos(2. * i * alpha)
y_proj: float = np.sin(2. * i * alpha)
r: float = rmax * (min_inv_m * y_proj + x_proj)[:, None, None] / (min_inv_m[:, None, None] * np.sin(phi) + np.cos(phi))

dist: float = (rho - r)

# +
fig, axes = plt.subplots(1, n, figsize=(n * 4, 3))
for _i in i:
   axes[_i].set_title("$r$")
   _map = axes[_i].imshow(dist[_i], vmax=50, vmin=-50)
   fig.colorbar(_map, ax=axes[_i])
plt.show()    

fig, axes = plt.subplots(1, n, figsize=(n * 4, 3))
for _i in i:
   axes[_i].set_title("$r$")
   _map = axes[_i].imshow(dist[_i] * wedge[_i])
   fig.colorbar(_map, ax=axes[_i])
plt.show()
# -

dist: float = (dist * wedge).sum(axis=0)
amax: callable = lambda arr: np.abs(arr).max()
smooth: callable = lambda arr: .5 * (np.tanh(npix * arr) + 1.)

# +
fig = plt.figure()
axes = plt.axes()
_map = axes.imshow(dist, cmap=plt.cm.seismic, vmin=-amax(dist), vmax=amax(dist))
fig.colorbar(_map, ax=axes)
plt.show()


fig = plt.figure()
axes = plt.axes()
_map = axes.imshow(smooth(dist))
fig.colorbar(_map, ax=axes)
plt.show()
# -

# # Vertex Generation of Polygons.
# So this is very challenging. I have made extensive notes but little progress. 
# OK so attempting to generate the vertices for a square. 
# This is going to give me infinite values. 

vertices: float = np.array([[.5, .5], [.5, -.5], [-.5, -.5], [-.5, .5]], float)
vertices: float = np.tile(vertices, (1000, 1))


def calc_edge_grad_from_vert(vertices: float) -> float:
    diffs: float = vertices - np.roll(vertices, (1, 1))
    return diffs[:, 1] / diffs[:, 0]


m: float = calc_edge_grad_from_vert(vertices)[:, None, None]

    y1: float = vertices[:, 1][:, None, None]


def slice_v1()
    x1: float = vertices[:, 0][:, None, None]

    return x1


# %%timeit
x1: float = vertices[:, 0]
y1: float = vertices[:, 1]

x: float = coords[0][None, :, :]
y: float = coords[1][None, :, :]


@jax.jit
def perp_dist_from_line(m: float, x1: float, y1: float, x: float, y: float) -> float:
    inf_case: float = (x - x1)
    gen_case: float = (m * (x - x1) - (y - y1)) / np.sqrt(1 + m ** 2)
    return np.where(np.isinf(m), inf_case, gen_case)


theta: float = np.arctan2(y1, x1)

two_pi: float = 2. * np.pi
offset_theta: float = offset(theta, 0.)


@jax.jit
def offset(theta: float, threshold: float) -> float:
    comps: float = np.array(theta < threshold, float)
    return theta + comps * two_pi


sorted_inds: int = np.argsort(offset_theta)

sorted_x1: float = x1[sorted_inds]
sorted_y1: float = y1[sorted_inds]
sorted_theta: float = offset_theta[sorted_inds]
next_sorted_theta: float = np.roll(sorted_theta, -1).at[-1].add(two_pi)
sorted_m: float = m[sorted_inds]

d: float = perp_dist_from_line(sorted_m, sorted_x1, sorted_y1, x, y)  

bc_phi: float = phi[None, :, :]
bc_sort_theta: float = sorted_theta[:, None, None]
bc_next_sort_theta: float = next_sorted_theta[:, None, None]

phi: float = offset(np.arctan2(y, x), sorted_theta[0])
w: float = ((bc_phi >= bc_sort_theta) & (bc_phi < bc_next_sort_theta)).astype(float)

# +
fig, axes = plt.subplots(3, 4, figsize=(4*4, 9))

for i in range(4):
    _map = axes[0][i].imshow(w[i, :, :])
    fig.colorbar(_map, ax=axes[0][i])
    
    _map = axes[1][i].imshow(d[i, :, :])
    fig.colorbar(_map, ax=axes[1][i])
    
    _map = axes[2][i].imshow(d[i, :, :] * w[i, :, :])
    fig.colorbar(_map, ax=axes[2][i])
# -


polygon: float = (dist_sgn * d * w).sum(axis=0)

plt.imshow(polygon)
plt.colorbar()

# +
# I should just be able to multiply by the sign of the distance from zero 
# -

dist_sgn: int = np.sign(perp_dist_from_line(sorted_m, sorted_x1, sorted_y1, np.array([[0.]]), np.array([[0.]])))

dist_sgn.shape

dist_sgn


