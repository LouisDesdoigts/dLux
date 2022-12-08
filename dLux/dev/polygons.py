import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "seismic"

n: int = 5
alpha: float = np.pi / n # Half the angular disp of one wedge

grid: float = np.linspace(0, 1., 100) - .5
coords: float = np.meshgrid(grid, grid)

neg_pi_to_pi_phi: float = np.arctan2(coords[1], coords[0]) 
phi: float = neg_pi_to_pi_phi + 2. * (neg_pi_to_pi_phi < 0.) * np.pi
r: float = np.hypot(coords[0], coords[1])

i: int = 2
low_bound: float = 2. * i * alpha
top_bound: float = 2. * (i + 1.) * alpha

wedge = ((low_bound < phi) & (phi < top_bound)).astype(float)

abs_max = lambda arr: np.abs(arr).max()

plt.title(f"$\\phi_{i}$")
plt.imshow(wedge, vmin=-abs_max(wedge), vmax=abs_max(wedge))
plt.colorbar()
plt.show()

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


