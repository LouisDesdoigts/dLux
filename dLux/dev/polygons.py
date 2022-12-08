import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "seismic"

n: int = 5
alpha: float = np.pi / n # Half the angular disp of one wedge

grid: float = np.linspace(0, 1., 100) - .5
coords: float = np.meshgrid(grid, grid)

phi: float = np.arctan2(coords[1], coords[0])
r: float = np.hypot(coords[0], coords[1])

abs_max = lambda arr: np.abs(arr).max()
plt.title("$r$")
plt.imshow(r, vmin=-abs_max(r), vmax=abs_max(r))
plt.colorbar()
plt.show()

plt.title("$\\phi$")
plt.imshow(phi, vmin=-abs_max(phi), vmax=abs_max(phi))
plt.colorbar()
plt.show()


