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

from apertures import DynamicAperture
from typing import TypeVar
from abc import ABC

ApertureLayer = TypeVar("ApertureLayer")
Array = TypeVar("Array")


class 


class PolygonalAperture(DynamicAperture, ABC):
    """
    An abstract class that represents all `PolygonalApertures`.
    The structure here is more than a little strange. Most of 
    the pre-implemented `PolygonalApertures` do **not** inherit
    from `PolygonalAperture`. This is because most of the
    behaviour that is defined by `PolygonalAperture` is related
    to general cases. For apertures, the generality results in 
    a loss of speed. For example, this may be caused because
    a specific symmetry of the shape cannot be exploited. As 
    a result, more optimal implementations could be created 
    directly. Since, the pre-implemented `Aperture` classes 
    that are polygonal share no behaviour with the 
    `PolygonalAperture` it made more sense to separate them 
    out. 

    Parameters
    ----------
    centre: float, meters
        The centre of the coordinate system along the x-axis.
    softening: bool = False
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool = False
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """
    
    def __init__(self   : ApertureLayer, 
            centre      : Array = [0., 0.], 
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer:
        """
        Parameters
        ----------
        centre: float, meters
            The centre of the coordinate system along the x-axis.
        softening: bool = False
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool = False
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: float, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(
            centre = centre, 
            strain = strain, 
            comression = compression,
            rotation = rotation,
            occulting = occulting,
            softening = softening)
    
    
    def _perp_dists_from_lines(
            self: ApertureLayer, 
            m   : float, 
            x1  : float, 
            y1  : float,
            x   : float, 
            y   : float) -> float:
        """
        Calculate the perpendicular distance of a set of points (x, y) from
        a line parametrised by a gradient m and a point (x1, y1). Notice, 
        I am using x and y separately because the instructions cannot be vectorised
        accross them combined. This function can take any number of points.
        
        Parameters:
        -----------
        m: float, None (meters / meter)
            The gradient of the line.
        x1: float, meters
            The x coordinate of a single point that lies on the line.
        y1: float, meters
            The y coordinate of a single point that lies on the line. 
        x: float, meters
            A set of coordinates that you wish to calculate the distance to 
            from the line. 
        y: float, meters
            A set of coordinates that you wish to calculate the distance to 
            from the line. Must have the same dimensions as x.
        
        Returns:
        --------
        dists: float, meters
            The distance of the points (x, y) from the line. Has the same 
            shape as x and y.
        """
        inf_case: float = (x - x1)
        gen_case: float = (m * inf_case - (y - y1)) / np.sqrt(1 + m ** 2)
        return np.where(np.isinf(m), inf_case, gen_case)
    
    
    def _grad_from_two_points(
            self: ApertureLayer, 
            xs  : float, 
            ys  : float) -> float:
        """
        Calculate the gradient of the chord that connects two points. 
        Note: This is distinct from `_grads_from_many_points` in that
        it does not wrap arround.
        
        Parameters:
        -----------
        xs: float, meters
            The x coordinates of the points.
        ys: float, meters
            The y coordinates of the points.
            
        Returns:
        --------
        m: float, None (meters / meter)
            The gradient of the chord that connects the two points.
        """
        return (ys[1] - ys[0]) / (xs[1] - xs[0])
    
    
    def _offset(
            self        : ApertureLayer, 
            theta       : float, 
            threshold   : float) -> float:
        """
        Transform the angular range of polar coordinates so that 
        the new lowest angle is offset. The final range should be 
        $[\\phi, \\phi + 2 \\pi]$ where $\\phi$ represents the 
        `threshold`. 
        
        Parameters:
        -----------
        theta: float, radians
            The angular coordinates.
        threshold: float
            The amount to offset the coordinates by.
        
        Returns:
        --------
        theta: float, radians 
            The offset coordinate system.
        """
        comps: float = (theta < threshold).astype(float)
        return theta + comps * two_pi
    
    
    def _is_orig_left_of_edge(
            self: ApertureLayer, 
            ms  : float, 
            xs  : float, 
            ys  : float) -> int:
        """
        Determines whether the origin is to the left or the right of 
        the edge. The edge(s) in this case are defined by a set of 
        gradients, m and points (xs, ys).
        
        Parameters:
        -----------
        ms: float, None (meters / meter)
            The gradient of the edge(s).
        xs: float, meters
            A set of x coordinates that lie along the edges. 
            Must have the same shape as ms. 
        ys: float, meters
            A set of y coordinates that lie along the edges.
            Must have the same shape as ms.
            
        Returns:
        --------
        is_left: int
            1 if the origin is to the left else -1.
        """
        bc_orig: float = np.array([[0.]])
        dist_from_orig: float = perp_dist_from_line(sm, sx1, sy1, bc_orig, bc_orig)
        return np.sign(dist_from_orig)
    
    
    def _make_wedges(off_phi: float, sorted_theta: float) -> float:
        """
        Wedges are used to isolate the space between two vertices in the 
        angular plane. 
        
        Parameters:
        -----------
        off_phi: float, radians
            The angular coordinates that have been correctly offset so 
            that the minimum angle corresponds to the first vertex.
            Note that this particular offset is not unique as any offset
            that is two pi greater will also work.
        sorted_theta: float, radians
            The angles of the vertices sorted from lowest to highest. 
            Implementation Note: The sorting is required for other 
            functions that are typically called together. As a result 
            it has not been internalised. This is a helper function 
            that is not designed to be called in general. 
            
        Returns:
        --------
        wedges: float
            A stack of binary (float) arrays that represent the angles 
            bounded by each consecutive pair of vertices.
        """
        next_sorted_theta: float = np.roll(sorted_theta, -1).at[-1].add(two_pi)
        bc_next_sort_theta: float = next_sorted_theta
        greater_than: bool = (off_phi >= sorted_theta)
        less_than: bool = (off_phi < bc_next_sort_theta)
        wedges: bool = greater_than & less_than
        return wedges.astype(float)


class IrregularPolygonalAperture(PolygonalAperture):
    """
    The default aperture is dis-allows the learning of all 
    parameters. 

    Parameters
    ----------
    centre: float, meters
        The centre of the coordinate system along the x-axis.
    softening: bool = False
        True if the aperture is soft edged otherwise False. A
        soft edged aperture has a small layer of non-binary 
        pixels. This is to prevent undefined gradients. 
    occulting: bool = False
        True if the aperture is occulting else False. An 
        occulting aperture is zero inside and one outside. 
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    vertices: Array, meters
        The location of the vertices of the aperture.
    """
    vertices: Array
    
    
    def __init__(self   : ApertureLayer, 
            vertices    : Array,
            centre      : Array = [0., 0.], 
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : bool = False) -> ApertureLayer:
        """
        Parameters
        ----------
        vertices: Array, meters
            The location of the vertices of the aperture.
        centre: float, meters
            The centre of the coordinate system along the x-axis.
        softening: bool = False
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool = False
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: float, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(
            centre = centre, 
            strain = strain, 
            comression = compression,
            rotation = rotation,
            occulting = occulting,
            softening = softening)
        self.vertices = np.array(vertices).astype(float)

    
    
    def _grads_from_many_points(x1: float, y1: float) -> float:
        """
        Given a set of points, calculate the gradient of the line that 
        connects those points. This function assumes that the points are 
        provided in the order they are to be connected together. Notice 
        that we also assume there are more than two points, but more can 
        be provided in which case the shape is assumed to be closed. The 
        output has the same shape as the input and does not check for 
        infinite (vertical) gradients.
        
        Due to the intensly vectorised nature of this code it is ofen 
        necessary to provided the parameters with expanded dimensions. 
        This may be achieved using `x1[:, None, None]` or 
        `x1.reshape((-1, 1, 1))` or `np.expand_dims(x1, (1, 2))`.
        There is no major performance difference between the different
        methods of reshaping. 
        
        Parameters:
        -----------
        x1: float, meters
            The x coordinates of the points that are to be connected. 
        y1: float, meters
            The y coordinates of the points that are to be connected. 
            Must have the same shape as x. 
            
        Returns:
        --------
        ms: float, None (meters / meter)
            The gradients of the lines that connect the vertices. The 
            vertices wrap around to form a closed shape whatever it 
            may look like. 
        """
        x_diffs: float = x1 - np.roll(x1, -1)
        y_diffs: float = y1 - np.roll(y1, -1)
        return y_diffs / x_diffs
    

@jax.jit
def draw_from_vertices(vertices: float, coords: float) -> float:
    two_pi: float = 2. * np.pi
    
    bc_x1: float = vertices[:, 0][:, None, None]
    bc_y1: float = vertices[:, 1][:, None, None]

    bc_x: float = coords[0][None, :, :]
    bc_y: float = coords[1][None, :, :]
        
    theta: float = np.arctan2(bc_y1, bc_x1)
    offset_theta: float = offset(theta, 0.)
        
    sorted_inds: int = np.argsort(offset_theta.flatten())
        
    sorted_x1: float = bc_x1[sorted_inds]
    sorted_y1: float = bc_y1[sorted_inds]
    sorted_theta: float = offset_theta[sorted_inds]   
    sorted_m: float = calc_edge_grad_from_vert(sorted_x1, sorted_y1)
        
    phi: float = offset(np.arctan2(bc_y, bc_x), sorted_theta[0])
           
    dist_from_edges: float = perp_dist_from_line(sorted_m, sorted_x1, sorted_y1, bc_x, bc_y)  
    wedges: float = make_wedges(phi, sorted_theta)
    dist_sgn: float = is_inside(sorted_m, sorted_x1, sorted_y1)
        
    return (dist_sgn * dist_from_edges * wedges).sum(axis=0)



# # Testing against different scenarios
#
# The point of this will be to test many regular polygons as well as a few aditional shapes.

smooth: callable = jax.jit(smooth)

two_pi: float = 2. * np.pi


def reg_pol_verts(n: int, r: float) -> float:
    thetas: float = np.linspace(0., two_pi, n, endpoint=False)
    return np.transpose(r * np.array([np.cos(thetas), np.sin(thetas)]))


sq_verts: float = reg_pol_verts(4, .5)
pent_verts: float = reg_pol_verts(5, .5)
hex_verts: float = reg_pol_verts(6, .5)

rand_verts: float = np.array([[.5, .5], [-.3, .4], [0., -.2], [.2, -.1], [.5, -.5]])

hexagon: float = draw_from_vertices(hex_verts, coords)
pentagon: float = draw_from_vertices(pent_verts, coords)
square: float = draw_from_vertices(sq_verts, coords)
rand: float = draw_from_vertices(rand_verts, coords)

fig, axes = plt.subplots(1, 4, figsize=(4*4, 3))
cmap = axes[0].imshow(smooth(hexagon))
fig.colorbar(cmap, ax=axes[0])
cmap = axes[1].imshow(smooth(pentagon))
fig.colorbar(cmap, ax=axes[1])
cmap = axes[2].imshow(smooth(square))
fig.colorbar(cmap, ax=axes[2])
cmap = axes[3].imshow(smooth(rand))
fig.colorbar(cmap, ax=axes[3])

# # Testing against alternate implementations
#
# This is testing against my pre-existing simple square implementation. 

# %%timeit
polygon: float = smooth(draw_from_vertices(sq_verts, coords))

polygon_v1: float = smooth(draw_from_vertices(sq_verts, coords))


@jax.jit
def simp_square(coords: float, width: float) -> float:
    mask: float = - np.abs(coords) + width / 2.       
    return np.prod(smooth(mask), axis=0)


coords: float = np.array(coords)

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


