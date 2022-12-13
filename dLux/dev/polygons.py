import jax
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from apertures import DynamicAperture
from typing import TypeVar
from abc import ABC

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "inferno"

# # Vertex Generation of Polygons.
# So this is very challenging. I have made extensive notes but little progress. 
# OK so attempting to generate the vertices for a square. 
# This is going to give me infinite values. 
#
# A note on conventions. I am using `bc` to represent broadcastable. This is just a copy that has had expanded dimensions ect.
#
# Hang on: I think that I just worked out a better way to do this. If I can generate the distance from a line parallel to the edge and passing through the origin then I just need to subtract the distance to the edge from the origin. I will finish the current implementation and then I will try this. 

ApertureLayer = TypeVar("ApertureLayer")
Array = TypeVar("Array")


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
    
    Implementation Notes: A lot of the code that is provided 
    was carefully hand vectorised. In general, where a shape 
    change is applied to an array the new array is given the 
    prefix `bc` standing for "broadcastable".

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
            compression = compression,
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
        dist_from_orig: float = self._perp_dists_from_lines(ms, xs, ys, bc_orig, bc_orig)
        return np.sign(dist_from_orig)
    
    
    def _make_wedges(self: ApertureLayer, off_phi: float, sorted_theta: float) -> float:
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
            that is not designed to be called in general. This should 
            have the correct shape to be braodcast. This usually involves 
            expanding it to have two extra dimensions. 
            
        Returns:
        --------
        wedges: float
            A stack of binary (float) arrays that represent the angles 
            bounded by each consecutive pair of vertices.
        """
        next_sorted_theta: float = np.roll(sorted_theta, -1).at[-1].add(two_pi)
        greater_than: bool = (off_phi >= sorted_theta)
        less_than: bool = (off_phi < next_sorted_theta)
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
            compression = compression,
            rotation = rotation,
            occulting = occulting,
            softening = softening)
        self.vertices = np.array(vertices).astype(float)

    
    
    def _grads_from_many_points(self: ApertureLayer, x1: float, y1: float) -> float:
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
    
    
    def _extent(self: ApertureLayer) -> float:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre. For inherited classes, consider implementing analytically for speed.

        Parameters
        ----------
        coordinates : Array
            The cartesian coordinates to generate the hexikes on.
            The dimensions of the tensor should be `(2, npix, npix)`.
            where the leading axis is the x and y dimensions.  

        Returns
        -------
        extent : float
            The maximum distance from centre to edge of aperture
        """
        verts: float = self.vertices
        dist_to_verts: float = np.hypot(verts[:, 1], verts[:, 0])
        return np.max(dist_to_verts)
    
    
    def _metric(self: ApertureLayer, coords: float) -> float:
        """
        A measure of how far a pixel is from the aperture.
        This is a very abstract description that was constructed 
        when dealing with the soft edging. For a normal binary 
        representation the metric is zero if it is inside the
        aperture and one if it is outside the aperture. Notice,
        we have not attempted to prove that this is a metric 
        via the axioms, this is just a handy name that brings 
        to mind the general idea. For a soft edged aperture the 
        metric is different.

        Parameters:
        -----------
        distances: Array
            The distances of each pixel from the edge of the aperture. 
            Again, the words distances is designed to aid in 
            conveying the idea and is not strictly true. We are
            permitting negative distances when inside the aperture
            because this was simplest to implement. 

        Returns:
        --------
        non_occ_ap: Array 
            This is essential the final step in processing to produce
            the aperture. What is returned is the non-occulting 
            version of the aperture. 
        """
        two_pi: float = 2. * np.pi

        bc_x1: float = self.vertices[:, 0][:, None, None]
        bc_y1: float = self.vertices[:, 1][:, None, None]

        bc_x: float = coords[0][None, :, :]
        bc_y: float = coords[1][None, :, :]

        theta: float = np.arctan2(bc_y1, bc_x1)
        offset_theta: float = self._offset(theta, 0.)

        sorted_inds: int = np.argsort(offset_theta.flatten())

        sorted_x1: float = bc_x1[sorted_inds]
        sorted_y1: float = bc_y1[sorted_inds]
        sorted_theta: float = offset_theta[sorted_inds]   
        sorted_m: float = self._grads_from_many_points(sorted_x1, sorted_y1)

        phi: float = self._offset(np.arctan2(bc_y, bc_x), sorted_theta[0])

        dist_from_edges: float = self._perp_dists_from_lines(sorted_m, sorted_x1, sorted_y1, bc_x, bc_y)  
        wedges: float = self._make_wedges(phi, sorted_theta)
        dist_sgn: float = self._is_orig_left_of_edge(sorted_m, sorted_x1, sorted_y1)

        flat_dists: float = (dist_sgn * dist_from_edges * wedges).sum(axis=0)
        return self._soften(flat_dists)


class RegularPolygonalAperture(PolygonalAperture):
    """
    An optiisation that can be applied to generate
    regular polygonal apertures without using their 
    vertices. 
    
    Parameters:
    -----------
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
    nsides: int
        The number of sides that the aperture has. 
    rmax: float, meters
        The radius of the smallest circle that can completely 
        enclose the aperture. 
    """
    nsides: int
    rmax: float
        
    
    def __init__(self   : ApertureLayer, 
            nsides      : int,
            rmax        : float,
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
            compression = compression,
            rotation = rotation,
            occulting = occulting,
            softening = softening)
        self.nsides = int(nsides)
        self.rmax = np.array(rmax).astype(float)
        
        
    def _extent(self: ApertureLayer) -> float:
        """
        Returns the largest distance to the outer edge of the aperture from the
        centre. For inherited classes, consider implementing analytically for speed.

        Parameters
        ----------
        coordinates : Array
            The cartesian coordinates to generate the hexikes on.
            The dimensions of the tensor should be `(2, npix, npix)`.
            where the leading axis is the x and y dimensions.  

        Returns
        -------
        extent : float
            The maximum distance from centre to edge of aperture
        """
        return self.rmax
        
    
    def _metric(self: ApertureLayer, coords: float) -> float:
        """
        A measure of how far a pixel is from the aperture.
        This is a very abstract description that was constructed 
        when dealing with the soft edging. For a normal binary 
        representation the metric is zero if it is inside the
        aperture and one if it is outside the aperture. Notice,
        we have not attempted to prove that this is a metric 
        via the axioms, this is just a handy name that brings 
        to mind the general idea. For a soft edged aperture the 
        metric is different.

        Parameters:
        -----------
        distances: Array
            The distances of each pixel from the edge of the aperture. 
            Again, the words distances is designed to aid in 
            conveying the idea and is not strictly true. We are
            permitting negative distances when inside the aperture
            because this was simplest to implement. 

        Returns:
        --------
        non_occ_ap: Array 
            This is essential the final step in processing to produce
            the aperture. What is returned is the non-occulting 
            version of the aperture. 
        """
        x: float = coords[0]
        y: float = coords[1]

        neg_pi_to_pi_phi: float = np.arctan2(y, x) 
        alpha: float = np.pi / self.nsides
            
        i: int = np.arange(self.nsides)[:, None, None] # Dummy index
        bounds: float = 2. * i * alpha + alpha
        phi: float = self._offset(neg_pi_to_pi_phi, bounds[0])
            
        wedges: float = self._make_wedges(phi, bounds)
        ms: float = -1 / np.tan(2. * (i + 1.) * alpha)
        xs: float = self.rmax * np.cos(2. * i * alpha)
        ys: float = self.rmax * np.sin(2. * i * alpha)
        dists: float = self._perp_dists_from_lines(ms, xs, ys, x, y)
        inside: float = self._is_orig_left_of_edge(ms, xs, ys)
         
        dist: float = (inside * dists * wedges)
            
        print("M: ", ms)
        print("XS: ", xs)
        print("YS: ", ys)
        
        fig, axes = plt.subplots(1, self.nsides, figsize=(self.nsides*4, 1*3))
        for i in np.arange(self.nsides):
            cmap = axes[i].imshow(dists[i])
            fig.colorbar(cmap, ax=axes[i])
        plt.show()
        
        plt.imshow(dist.sum(axis=0))
        plt.colorbar()
        plt.show()


        return self._soften(dist.sum(axis=0))

# # Testing against different scenarios
#
# The point of this will be to test many regular polygons as well as a few aditional shapes.

npix: int = 100
grid: float = np.linspace(0, 2., npix) - 1.
coords: float = np.array(np.meshgrid(grid, grid))
two_pi: float = 2. * np.pi


# ## IrregularPolygonalAperture

def reg_pol_verts(n: int, r: float) -> float:
    thetas: float = np.linspace(0., two_pi, n, endpoint=False)
    return np.transpose(r * np.array([np.cos(thetas), np.sin(thetas)]))


sq_verts: float = reg_pol_verts(4, .5)
pent_verts: float = reg_pol_verts(5, .5)
hex_verts: float = reg_pol_verts(6, .5)
rand_verts: float = np.array([[.5, .5], [-.3, .4], [0., -.2], [.2, -.1], [.5, -.5]])

# +
sq_ireg_aper: ApertureLayer = IrregularPolygonalAperture(sq_verts)
hex_ireg_aper: ApertureLayer = IrregularPolygonalAperture(hex_verts)
pent_ireg_aper: ApertureLayer = IrregularPolygonalAperture(pent_verts)
rand_ireg_aper: ApertureLayer = IrregularPolygonalAperture(rand_verts)

sq_aper: float = sq_ireg_aper._aperture(coords)
hex_aper: float = hex_ireg_aper._aperture(coords)
pent_aper: float = pent_ireg_aper._aperture(coords)
rand_aper: float = rand_ireg_aper._aperture(coords)

fig, axes = plt.subplots(1, 4, figsize=(4*4, 3))
cmap = axes[0].imshow(sq_aper)
fig.colorbar(cmap, ax=axes[0])
cmap = axes[1].imshow(pent_aper)
fig.colorbar(cmap, ax=axes[1])
cmap = axes[2].imshow(hex_aper)
fig.colorbar(cmap, ax=axes[2])
cmap = axes[3].imshow(rand_aper)
fig.colorbar(cmap, ax=axes[3])
# -
# ## RegularPolygonalAperture

amax: callable = lambda arr: np.abs(arr).max()


# +
sq_reg_aper: ApertureLayer = RegularPolygonalAperture(4, 1.)
hex_reg_aper: ApertureLayer = RegularPolygonalAperture(6, 1.)
pent_reg_aper: ApertureLayer = RegularPolygonalAperture(5, 1.)

sq_aper: float = sq_reg_aper._aperture(coords)
hex_aper: float = hex_reg_aper._aperture(coords)
pent_aper: float = pent_reg_aper._aperture(coords)

fig, axes = plt.subplots(1, 3, figsize=(3*4, 3))
cmap = axes[0].imshow(sq_aper)
fig.colorbar(cmap, ax=axes[0])
cmap = axes[1].imshow(pent_aper)
fig.colorbar(cmap, ax=axes[1])
cmap = axes[2].imshow(hex_aper)
fig.colorbar(cmap, ax=axes[2])
# -



