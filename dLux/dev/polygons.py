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

#fig, axes = plt.subplots(1, n, figsize=(n * 4, 3))
#for _i in i:
#    axes[_i].set_title("$r$")
#    _map = axes[_i].imshow(dist[_i], vmax=50, vmin=-50)
#    fig.colorbar(_map, ax=axes[_i])
#plt.show()    
#
#fig, axes = plt.subplots(1, n, figsize=(n * 4, 3))
#for _i in i:
#    axes[_i].set_title("$r$")
#    _map = axes[_i].imshow(dist[_i] * wedge[_i])
#    fig.colorbar(_map, ax=axes[_i])
#plt.show()

dist: float = (dist * wedge).sum(axis=0)
amax: callable = lambda arr: np.abs(arr).max()
smooth: callable = lambda arr: .5 * (np.tanh(npix * arr) + 1.)

#fig = plt.figure()
#axes = plt.axes()
#_map = axes.imshow(dist, cmap=plt.cm.seismic, vmin=-amax(dist), vmax=amax(dist))
#fig.colorbar(_map, ax=axes)
#plt.show()
#
#
#fig = plt.figure()
#axes = plt.axes()
#_map = axes.imshow(smooth(dist))
#fig.colorbar(_map, ax=axes)
#plt.show()

# OK so attempting to generate the vertices for a square. 
# this is going to give me infinite values. 
vertices: float = np.array([[.5, .5], [.5, -.5], [-.5, -.5], [-.5, .5]], float)
diffs: float = vertices - np.roll(vertices, (1, 1))
m: float = diffs[:, 1] / diffs[:, 0]

x1: float = vertices[:, 0]
y1: float = vertices[:, 1]

x: float = coords[0][:, :, None]
y: float = coords[1][:, :, None]

d: float = np.abs(m * (x - x1) - (y - y1)) / np.sqrt(1 + m ** 2)
theta: float = np.arctan2(y1, x1)

phi: float = np.arctan2(y, x)
# TODO: From here I need to get this working. 
w: float = ((phi < theta) & (phi > np.roll(theta, 1))).astype(float)

plt.imshow(w[:, :, 0])
plt.colorbar()
plt.show()
print(theta)

