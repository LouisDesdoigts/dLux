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
#
# A note on conventions. I am using `bc` to represent broadcastable. This is just a copy that has had expanded dimensions ect.
#
# Hang on: I think that I just worked out a better way to do this. If I can generate the distance from a line parallel to the edge and passing through the origin then I just need to subtract the distance to the edge from the origin. I will finish the current implementation and then I will try this. 

def draw_from_vertices(vertices: float, coords: float) -> float:
    two_pi: float = 2. * np.pi
    
    bc_m: float = calc_edge_grad_from_vert(vertices)[:, None, None]
    
    bc_x1: float = vertices[:, 0][:, None, None]
    bc_y1: float = vertices[:, 1][:, None, None]

    bc_x: float = coords[0][None, :, :]
    bc_y: float = coords[1][None, :, :]
        
    theta: float = np.arctan2(y1, x1)
    offset_theta: float = offset(theta, 0.)
        
    sorted_inds: int = np.argsort(offset_theta.flatten())
        
    sorted_x1: float = bc_x1[sorted_inds]
    sorted_y1: float = bc_y1[sorted_inds]
    sorted_m: float = bc_m[sorted_inds]
    sorted_theta: float = offset_theta[sorted_inds]
        
    phi: float = offset(np.arctan2(y, x), sorted_theta[0])
        
    dist_from_edges: float = perp_dist_from_line(sorted_m, sorted_x1, sorted_y1, x, y)  
    wedges: float = make_wedges(phi, sorted_theta)
    dist_sgn: float = is_inside(sorted_m, sorted_x1, sorted_y1)
        
    fig, axes = plt.subplots(1, 6, figsize=(6*4, 3))
    for i in range(6):
        cmap = axes[i].imshow(wedges[i])
        fig.colorbar(cmap, ax=axes[i])
    plt.show() 
        
    return (dist_sgn * dist_from_edges * wedges).sum(axis=0)


def calc_edge_grad_from_vert(vertices: float) -> float:
    diffs: float = vertices - np.roll(vertices, (1, 1))
    return diffs[:, 1] / diffs[:, 0]


def perp_dist_from_line(m: float, x1: float, y1: float, x: float, y: float) -> float:
    inf_case: float = (x - x1)
    gen_case: float = (m * inf_case - (y - y1)) / np.sqrt(1 + m ** 2)
    return np.where(np.isinf(m), inf_case, gen_case)


def offset(theta: float, threshold: float) -> float:
    comps: float = (theta < threshold).astype(float)
    return theta + comps * two_pi


def is_inside(sm: float, sx1: float, sy1) -> int:
    bc_orig: float = np.array([[0.]])
    dist_from_orig: float = perp_dist_from_line(sm, sx1, sy1, bc_orig, bc_orig)
    return np.sign(dist_from_orig)


def make_wedges(off_phi: float, sorted_theta: float) -> float:
    next_sorted_theta: float = np.roll(sorted_theta, -1).at[-1].add(two_pi)
    bc_next_sort_theta: float = next_sorted_theta
    greater_than: bool = (bc_phi >= bc_sort_theta)
    less_than: bool = (bc_phi < bc_next_sort_theta)
    wedges: bool = greater_than & less_than
    return wedges.astype(float)


# # Testing against different scenarios
#
# The point of this will be to test many regular polygons as well as a few aditional shapes.

def reg_pol_verts(n: int, r: float) -> float:
    thetas: float = np.linspace(0., two_pi, n, endpoint=False)
    return np.transpose(r * np.array([np.cos(thetas), np.sin(thetas)]))


hex_verts.shape

sq_verts: float = reg_pol_verts(4, .5)
pent_verts: float = reg_pol_verts(5, .5)
hex_verts: float = reg_pol_verts(6, .5)

polygon: float = draw_from_vertices(hex_verts, coords)

plt.imshow(polygon)
plt.colorbar()

# # Testing against alternate implementations
#
# This is testing against my pre-existing simple square implementation. 

# %%timeit
polygon: float = smooth(draw_from_vertices(vertices, coords))

polygon_v1: float = smooth(draw_from_vertices(vertices, coords))


@jax.jit
def simp_square(coords: float, width: float) -> float:
    mask: float = - np.abs(coords) + width / 2.       
    return np.prod(smooth(mask), axis=0)


# %%timeit
polygon: float = simp_square(coords, 1.)

polygon_v2: float = simp_square(coords, 1.)

fig, axes = plt.subplots(1, 3, figsize=(3*4, 3))
cmap = axes[0].imshow(polygon_v1)
fig.colorbar(cmap, ax=axes[0])
cmap = axes[1].imshow(polygon_v2)
fig.colorbar(cmap, ax=axes[1])
cmap = axes[2].imshow(polygon_v1 - polygon_v2)
fig.colorbar(cmap, ax=axes[2])

