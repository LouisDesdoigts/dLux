from jax import Array
import jax.numpy as np
import dLux.utils as dlu
from equinox import filter_jit as jit

__all__ = [
    "segmented_hex_cens",
    "non_redundant_support",
    "circular_aperture",
    "segmented_aperture",
    "sparse_aperture",
    "hst_like",
    "jwst_like",
    "euclid_like",
]


@jit
def _hex_cens(rmax: float) -> Array:
    """
    Returns the centres of the six neighbouring hexagons.

    Parameters
    ----------
    rmax : float
        The circumradius of each hexagon.

    Returns
    -------
    centers : Array
        The six neighbour centres with shape (6, 2).
    """
    r = np.sqrt(3.0) * rmax
    xys = []
    for i in range(6):
        angle_rad = dlu.deg2rad(60 * i + 30)
        xy = r * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        xys.append(xy)
    return np.array(xys)


@jit
def _evenly_spaced_points(point1: Array, point2: Array, n: int) -> Array:
    """
    Returns evenly spaced interior points between two 2D points.

    Parameters
    ----------
    point1 : Array
        The start point with shape (2,).
    point2 : Array
        The end point with shape (2,).
    n : int
        The number of interior points to return.

    Returns
    -------
    points : Array
        The interior points with shape (n, 2).
    """
    x = np.linspace(point1[0], point2[0], n + 2)[1:-1]
    y = np.linspace(point1[1], point2[1], n + 2)[1:-1]
    return np.squeeze(np.column_stack((x, y)))


@jit
def segmented_hex_cens(nrings: int, rmax: float, gap: float = 0.0) -> Array:
    """
    Hex-segment centres including the central segment.

    Parameters
    ----------
    nrings : int
        Number of segment rings including the central segment.
    rmax : float
        Circumradius of one hexagonal segment.
    gap : float = 0.0
        Physical gap between neighbouring segments.

    Returns
    -------
    centers : Array
        The segment centres with shape (n_segments, 2).

    Convention
    ----------
    nrings = 1 -> 1 segment
    nrings = 2 -> 7 segments
    nrings = 3 -> 19 segments
    """
    if nrings < 1:
        raise ValueError("`nrings` must be >= 1.")

    rseg = rmax + gap / np.sqrt(3.0)

    cens = [np.zeros((1, 2))]

    if nrings >= 2:
        inner = _hex_cens(rseg)
        cens.append(inner)

    for i in range(2, nrings):
        outer = _hex_cens(i * rseg)
        cens.append(outer)

        mids = []
        for j in range(len(outer)):
            m = _evenly_spaced_points(outer[j], outer[(j + 1) % 6], i - 1)
            mids.append(m)

        shaped = np.array(mids).reshape([6 * (i - 1), 2])
        cens.append(shaped)

    return np.concatenate(cens)


def non_redundant_support(apertures: Array) -> Array:
    """
    Returns a non-redundant support mask for overlapping sub-apertures.

    Parameters
    ----------
    apertures : Array
        The per-segment aperture masks with shape (n_segments, npixels, npixels).

    Returns
    -------
    support : Array
        A boolean support mask with the same shape as ``apertures`` and no pixel
        assigned to more than one segment.
    """
    # Get the hexagonal support
    aper_support = apertures > 0
    support_sum = aper_support.sum(0)
    redundant_mask = support_sum > 1

    # Select our redundant pixels
    redundant_pix = apertures[:, redundant_mask]

    # Get the index of the hexagon with the maximum value for each redundant pixel
    argmax = np.argmax(redundant_pix, axis=0)

    # Build the non-redundant support, choosing pixels with the maximum value
    inds = np.arange(redundant_pix.shape[1])
    empty = np.zeros_like(redundant_pix, dtype=bool)
    nr_support = empty.at[argmax, inds].set(True)

    # Remove the redundant pixels and paste back the non-redundant pixels
    aper_support = np.where(redundant_mask, False, aper_support)
    return aper_support.at[:, redundant_mask].set(nr_support)


def circular_aperture(
    npixels: int,
    diameter: float,
    oversample: int = 5,
    secondary_diameter: float | None = None,
    spider_width: float = 0.0,
    spider_angles: list | tuple | Array | None = None,
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
) -> Array | tuple[Array, Array]:
    """
    Builds a static circular aperture.

    Parameters
    ----------
    npixels : int
        The output size of the aperture arrays.
    diameter : float
        The full aperture diameter.
    oversample : int = 5
        The oversampling factor used to build soft pixel edges.
    secondary_diameter : float | None = None
        Optional central obscuration diameter.
    spider_width : float = 0.0
        Optional spider vane width.
    spider_angles : list | tuple | Array | None = None
        Angles of spider vanes in degrees.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the Zernike basis diameter.

    Returns
    -------
    transmission
        Shape (npixels, npixels) if ``zernike_nolls is None``.
    transmission, basis
        If Zernikes are requested, basis has shape
        ``(nterms, npixels, npixels)``.
    """
    if spider_width > 0 and spider_angles is None:
        raise ValueError("`spider_angles` must be provided when `spider_width > 0`.")

    # Get the oversampled primary aperture
    coords = dlu.pixel_coords(npixels * oversample, diameter=diameter)
    layers = [dlu.circle(coords, diameter / 2)]

    # Add the secondary if requested
    if secondary_diameter is not None and secondary_diameter > 0:
        layers.append(dlu.circle(coords, secondary_diameter / 2, invert=True))

    # Add the spiders if requested
    if spider_width > 0 and spider_angles is not None:
        layers += [dlu.spider(coords, spider_width, spider_angles)]

    # Get the combined transmission of the layers
    transmission = dlu.combine(layers, oversample)

    # Return the transmission if no Zernike basis is requested
    if zernike_nolls is None:
        return transmission

    # Get the non-oversampled Zernike basis for the primary aperture
    coords = dlu.pixel_coords(npixels, diameter=diameter)
    z_diam = diameter * (1.0 + zernike_oversize)
    basis = dlu.zernike_basis(zernike_nolls, coords, z_diam)

    # Mask the basis by the primary aperture (not including secondary or spiders)
    support = dlu.downsample(layers[0], oversample) > 0
    basis = basis * support[None, ...]

    # Return the transmission and basis
    return transmission, basis


def segmented_aperture(
    npixels: int,
    diameter: float,
    nrings: int,
    flat_to_flat: float,
    gap: float = 0.0,
    oversample: int = 5,
    has_secondary: bool = True,
    spider_width: float = 0.0,
    spider_angles: list | tuple | Array | None = None,
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
) -> Array | tuple[Array, Array]:
    """
    Builds a static segmented hexagonal aperture.

    Parameters
    ----------
    npixels : int
        The output size of the aperture arrays.
    diameter : float
        The full aperture diameter.
    nrings : int
        The number of hexagonal rings including the center segment.
    flat_to_flat : float
        Flat-to-flat diameter of each segment.
    gap : float = 0.0
        Physical gap between neighbouring segments.
    oversample : int = 5
        The oversampling factor used to build soft pixel edges.
    has_secondary : bool = True
        If True, removes the central segment as a secondary obscuration.
    spider_width : float = 0.0
        Optional spider vane width.
    spider_angles : list | tuple | Array | None = None
        Angles of spider vanes in degrees.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the segment Zernike basis diameter.

    Returns
    -------
    transmission
        Shape (npixels, npixels) if ``zernike_nolls is None``.
    transmission, basis
        If Zernikes are requested, basis has shape
        ``(n_segments, nterms, npixels, npixels)``.

    Notes
    -----
    The per-segment Zernikes are circular Zernikes in each local segment frame,
    clipped only by the binary segment mask.
    """
    if spider_width > 0 and spider_angles is None:
        raise ValueError("`spider_angles` must be provided when `spider_width > 0`.")

    # Get the oversampled coordinates
    coords = dlu.pixel_coords(npixels * oversample, diameter=diameter)
    shift_fn = jit(lambda c: dlu.translate_coords(coords, c))

    # Get the segment centres
    rmax = flat_to_flat / np.sqrt(3.0)
    cens = segmented_hex_cens(nrings, rmax, gap)

    # Get the individual hexagonal segments
    hex_fn = jit(lambda c: dlu.reg_polygon(shift_fn(c), rmax, 6))

    # Avoid vmap to stop ram blowing up
    hexes = np.array([hex_fn(c) for c in cens])

    # Central segment becomes the secondary obstruction
    if has_secondary:
        hexes = hexes[1:]
        cens = cens[1:]

    # Add the spiders if requested
    if spider_width > 0 and spider_angles is not None:
        spiders = dlu.spider(coords, spider_width, spider_angles)
    else:
        spiders = 1.0

    # Get the combined transmission of the layers
    transmission = dlu.combine([hexes.sum(0), spiders], oversample)

    # Return the transmission if no Zernike basis is requested
    if zernike_nolls is None:
        return transmission

    # Get the non-oversampled Zernike basis for each segment
    coords = dlu.pixel_coords(npixels, diameter=diameter)
    shift_fn = jit(lambda c: dlu.translate_coords(coords, c))

    # Get the zernike generation function
    z_diam = np.sqrt(3) * flat_to_flat * (1.0 + zernike_oversize)
    z_fn = lambda c: dlu.zernike_basis(zernike_nolls, shift_fn(c), z_diam)

    # Get the downsampled segment masks and supports
    # seg_support = [dlu.downsample(hex, oversample) > 0 for hex in hexes]
    hexes = np.array([dlu.downsample(hex, oversample) for hex in hexes])
    seg_support = non_redundant_support(hexes)

    # Calculate the basis for each segment and mask by the segment shape
    basis = [z_fn(cen) * supp[None, ...] for cen, supp in zip(cens, seg_support)]

    # Return the transmission and basis
    return transmission, np.array(basis)


def sparse_aperture(
    npixels: int,
    diameter: float,
    centers: list | tuple | Array,
    hole_diameter: float,
    shape: str = "circle",
    oversample: int = 5,
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
) -> Array | tuple[Array, Array]:
    """
    Builds a static sparse aperture from explicit sub-aperture centres.

    Parameters
    ----------
    npixels : int
        The output size of the aperture arrays.
    diameter : float
        The full aperture diameter.
    centers : list | tuple | Array
        Sub-aperture centers with shape ``(n_apertures, 2)``.
    hole_diameter : float
        Diameter of each sparse sub-aperture.
    shape : {"circle", "hex"} = "circle"
        Shape used for each sparse sub-aperture.
    oversample : int = 5
        The oversampling factor used to build soft pixel edges.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the sub-aperture Zernike basis diameter.

    Returns
    -------
    transmission
        Shape (npixels, npixels) if ``zernike_nolls is None``.
    transmission, basis
        If Zernikes are requested, basis has shape
        ``(n_ap, nterms, npixels, npixels)``.
    """
    if shape not in {"circle", "hex"}:
        raise ValueError("`shape` must be either 'circle' or 'hex'.")

    # Get the oversampled coordinates
    coords = dlu.pixel_coords(npixels * oversample, diameter=diameter)
    shift_fn = jit(lambda c: dlu.translate_coords(coords, c))

    # Pick the sub-aperture shape function
    if shape == "circle":
        aperture_fn = jit(lambda c: dlu.circle(shift_fn(c), hole_diameter / 2))
    else:
        rmax = hole_diameter / np.sqrt(3.0)
        aperture_fn = jit(lambda c: dlu.reg_polygon(shift_fn(c), rmax, 6))

    # Get the individual sub-apertures
    centers = np.array(centers, float)
    apers = [aperture_fn(cen) for cen in centers]

    # Get the combined transmission of the layers
    transmission = dlu.combine(apers, oversample, use_sum=True)

    # Return the transmission if no Zernike basis is requested
    if zernike_nolls is None:
        return transmission

    # Get the non-oversampled Zernike basis for each sub-aperture
    coords = dlu.pixel_coords(npixels, diameter=diameter)
    shift_fn = jit(lambda c: dlu.translate_coords(coords, c))

    # Get the zernike basis function
    z_diam = hole_diameter * (1.0 + zernike_oversize)
    if shape == "hex":
        z_diam *= np.sqrt(3.0)
    z_fn = lambda c: dlu.zernike_basis(zernike_nolls, shift_fn(c), z_diam)

    # Get the downsampled sub aperture support
    segs = [dlu.downsample(aper, oversample) for aper in apers]

    # Calculate the basis for each sub-aperture and mask by the sub-aperture shape
    basis = [z_fn(cen) * (seg > 0)[None, ...] for cen, seg in zip(centers, segs)]

    # Return the transmission and basis
    return transmission, np.array(basis)


def hst_like(
    npixels: int,
    diameter: float = 2.4,
    oversample: int = 5,
    secondary_diameter: float = 0.305,
    spider_width: float = 0.038,
    spider_angles: list | tuple = (0, 90, 180, 270),
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
) -> Array | tuple[Array, Array]:
    """
    Builds an HST-like circular aperture model.

    Parameters
    ----------
    npixels : int
        The output size of the aperture arrays.
    diameter : float = 2.4
        The primary mirror diameter.
    oversample : int = 5
        The oversampling factor used to build soft pixel edges.
    secondary_diameter : float = 0.305
        The secondary obscuration diameter.
    spider_width : float = 0.038
        Width of the spider vanes.
    spider_angles : list | tuple = (0, 90, 180, 270)
        Spider vane angles in degrees.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the Zernike basis diameter.

    Returns
    -------
    transmission : Array
        The HST-like aperture transmission.
    transmission, basis : tuple[Array, Array]
        Returned when ``zernike_nolls`` is provided.
    """
    return circular_aperture(
        npixels=npixels,
        diameter=diameter,
        oversample=oversample,
        secondary_diameter=secondary_diameter,
        spider_width=spider_width,
        spider_angles=spider_angles,
        zernike_nolls=zernike_nolls,
        zernike_oversize=zernike_oversize,
    )


def jwst_like(
    npixels: int,
    diameter: float = 6.6,
    nrings: int = 3,
    flat_to_flat: float = 1.32,
    gap: float = 0.007,
    oversample: int = 5,
    has_secondary: bool = True,
    spider_width: float = 0.1,
    spider_angles: list | tuple = (30, 180, 330),
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
) -> Array | tuple[Array, Array]:
    """
    Builds a JWST-like segmented aperture model.

    Parameters
    ----------
    npixels : int
        The output size of the aperture arrays.
    diameter : float = 6.6
        The full primary diameter.
    nrings : int = 3
        The number of hexagonal rings including the central segment.
    flat_to_flat : float = 1.32
        Flat-to-flat segment diameter.
    gap : float = 0.007
        Segment spacing.
    oversample : int = 5
        The oversampling factor used to build soft pixel edges.
    has_secondary : bool = True
        If True, removes the central segment.
    spider_width : float = 0.1
        Width of the spider vanes.
    spider_angles : list | tuple = (30, 180, 330)
        Spider vane angles in degrees.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the Zernike basis diameter.

    Returns
    -------
    transmission : Array
        The JWST-like aperture transmission.
    transmission, basis : tuple[Array, Array]
        Returned when ``zernike_nolls`` is provided.
    """
    return segmented_aperture(
        npixels=npixels,
        diameter=diameter,
        nrings=nrings,
        flat_to_flat=flat_to_flat,
        gap=gap,
        oversample=oversample,
        has_secondary=has_secondary,
        spider_width=spider_width,
        spider_angles=spider_angles,
        zernike_nolls=zernike_nolls,
        zernike_oversize=zernike_oversize,
    )


def euclid_like(
    npixels: int,
    diameter: float = 1.21,
    oversample: int = 5,
    secondary_diameter: float = 0.395,
    spider_width: float = 0.012,
    spider_angles: list | tuple = (0, 120, 240),
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
) -> Array | tuple[Array, Array]:
    """
    Builds a Euclid-like circular aperture model.

    Parameters
    ----------
    npixels : int
        The output size of the aperture arrays.
    diameter : float = 1.21
        The primary mirror diameter.
    oversample : int = 5
        The oversampling factor used to build soft pixel edges.
    secondary_diameter : float = 0.395
        The secondary obscuration diameter.
    spider_width : float = 0.012
        Width of the spider vanes.
    spider_angles : list | tuple = (0, 120, 240)
        Spider vane angles in degrees.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the Zernike basis diameter.

    Returns
    -------
    aperture : Array
        The Euclid-like aperture transmission.
    aperture, basis : tuple[Array, Array]
        Returned when ``zernike_nolls`` is provided.

    Notes
    -----
    This is an approximation that captures the main Euclid aperture features.
    """
    # Get the oversampled aperture without spiders or zernikes
    aperture = circular_aperture(
        npixels=npixels * oversample,
        diameter=diameter,
        secondary_diameter=secondary_diameter,
    )

    # Get the coordinates for the spiders
    ap_coords = dlu.pixel_coords(npixels * oversample, diameter=diameter)

    # Get the generation functions
    spider_shift = np.array([secondary_diameter / 2 - spider_width / 2, diameter / 2])
    rot_fn = jit(lambda angle: dlu.rotate_coords(ap_coords, dlu.deg2rad(angle + 30)))
    shift_fn = jit(lambda angle: dlu.translate_coords(rot_fn(angle), spider_shift))
    rect_fn = jit(lambda c: dlu.rectangle(c, spider_width, diameter, invert=True))

    # Get the spider vanes
    spiders = [rect_fn(shift_fn(angle)) for angle in spider_angles]

    # Combine the aperture and spiders, and downsample back to npixels
    aperture = dlu.combine([aperture] + list(spiders), oversample)

    if zernike_nolls is None:
        return aperture

    # Get the basis
    _, basis = circular_aperture(
        npixels=npixels,
        diameter=diameter,
        oversample=oversample,
        secondary_diameter=secondary_diameter,
        zernike_nolls=zernike_nolls,
        zernike_oversize=zernike_oversize,
    )

    return aperture, basis
