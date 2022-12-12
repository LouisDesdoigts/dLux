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
diffs: float = vertices - np.roll(vertices, (1, 1))
m: float = diffs[:, 1] / diffs[:, 0]

x1: float = vertices[:, 0]
y1: float = vertices[:, 1]

x: float = coords[0]
y: float = coords[1]

vcond: callable = jax.vmap(jax.lax.cond, in_axes=(0, None, None, 0, 0, 0))


def dist_from_line(m: float, x1: float, y1: float, x: float, y: float) -> float:
    inf_case: callable = lambda m, x1, y1: np.abs(x - x1)
    gen_case: callable = lambda m, x1, y1: np.abs(m * (x - x1) - (y - y1)) / np.sqrt(1 + m ** 2)
    return vcond(np.isinf(m), inf_case, gen_case, m, x1, y1)


d:float = dist_from_line(m, x1, y1, x, y)

d: float = np.abs(m * (x - x1) - (y - y1)) / np.sqrt(1 + m ** 2)
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

phi: float = offset(np.arctan2(y, x), sorted_theta[0])
w: float = ((phi > sorted_theta) & (phi < next_sorted_theta)).astype(float)

# +
fig, axes = plt.subplots(1, 4, figsize=(4*4, 3))

for i in range(4):
    _map = axes[i].imshow(w[:, :, i])
    fig.colorbar(_map, ax=axes[i])


# +
fig, axes = plt.subplots(1, 4, figsize=(4*4, 3))

for i in range(4):
    _map = axes[i].imshow(d[i, :, :])
    fig.colorbar(_map, ax=axes[i])
# -



print(theta)



