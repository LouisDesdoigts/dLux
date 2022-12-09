import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "inferno"

n: int = 5
rmax: float = 1.
alpha: float = np.pi / n # Half the angular disp of one wedge

npix: int = 100
grid: float = np.linspace(0, 2., npix) - 1.
coords: float = np.meshgrid(grid, grid)

neg_pi_to_pi_phi: float = np.arctan2(coords[1], coords[0]) 
phi: float = neg_pi_to_pi_phi + 2. * (neg_pi_to_pi_phi < 0.) * np.pi
rho: float = np.hypot(coords[0], coords[1])

fig, axes = plt.subplots(1, 2, figsize=(2 * 4, 3))
axes[0].set_title("$\\rho$")
_map = axes[0].imshow(rho, cmap=plt.cm.inferno)
fig.colorbar(_map, ax=axes[0])
axes[1].set_title("$\\phi$")
_map = axes[1].imshow(phi, cmap=plt.cm.inferno)
fig.colorbar(_map, ax=axes[1])
plt.show()

i: int = np.arange(n)
low_bound: float = 2. * i * alpha
top_bound: float = 2. * (i + 1.) * alpha

wedge: float = ((low_bound[:, None, None] < phi) & (phi < top_bound[:, None, None])).astype(float)
min_inv_m: float = np.tan((2. * i + 1.) * alpha)
x_proj: float = np.cos(2. * i * alpha)
y_proj: float = np.sin(2. * i * alpha)
r: float = rmax * (min_inv_m * y_proj + x_proj)[:, None, None] / (min_inv_m[:, None, None] * np.sin(phi) + np.cos(phi))

dist: float = (rho - r)

fig, axes = plt.subplots(1, n, figsize=(n * 4, 3))
for _i in i:
    axes[_i].set_title("$r$")
    _map = axes[_i].imshow(dist[_i] * wedge[_i])
    fig.colorbar(_map, ax=axes[_i])
plt.show()

dist: float = (dist * wedge).sum(axis=0)
amax: callable = lambda arr: np.abs(arr).max()
smooth: callable = lambda arr: .5 * (np.tanh(npix * arr) + 1.)

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

# OK I am stuck again. Well the positive side is always on the right.
# At least that is for a vertical line. I am not so sure how to 
# discuss this for a line that is not nessecarily ... Actually I 
# think that I just do this with abs and then check that if 
# r < my_line then multiply by negative one.

# I can also just check that the line is negative on the inside 
# how? 

# So I need to decipher what this does.
# phi + alpha rotates the whole thing backwards so that zero is 
# now along the spoke connecting the bottom of the right-most 
# edge. By dividing by the angular width of the edge it is 
# then able to enumerate the edges. This implies that I 
# probably want to add pi so that I am working on the 
# range zero two pi. 


#plt.title("$r$")
#plt.imshow(r, vmin=-abs_max(r), vmax=abs_max(r))
#plt.colorbar()
#plt.show()


