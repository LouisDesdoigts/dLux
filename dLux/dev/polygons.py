import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "seismic"

n: int = 5
rmax: float = 1.
alpha: float = np.pi / n # Half the angular disp of one wedge

grid: float = np.linspace(0, 2., 100) - 1.
coords: float = np.meshgrid(grid, grid)

neg_pi_to_pi_phi: float = np.arctan2(coords[1], coords[0]) 
phi: float = neg_pi_to_pi_phi + 2. * (neg_pi_to_pi_phi < 0.) * np.pi
r: float = np.hypot(coords[0], coords[1])

i: int = np.arange(n)
low_bound: float = 2. * i * alpha
top_bound: float = 2. * (i + 1.) * alpha

x: float = coords[0]
wedge: float = ((low_bound[:, None, None] < phi) & (phi < top_bound[:, None, None])).astype(float)
dist_from_rad: float = rmax * np.cos(alpha)

print(dist_from_rad)

dist: float = x + dist_from_rad

# NOTE: The distance of the edge from the circle is going to be
#       given by the 

abs_max = lambda arr: np.abs(arr).max()

for _i in i:
    plt.title(f"$\\phi_{i}$")
    plt.imshow(wedge[_i], vmin=-abs_max(dist), vmax=abs_max(dist))
    plt.colorbar()
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


