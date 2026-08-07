"""Generate sampled telescope apertures and aperture supports."""

import equinox as eqx
from jax import Array, vmap
import jax.numpy as np
import dLux.utils as dlu

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


@eqx.filter_jit
def _hex_cens(rmax: float) -> Array:
    """Returns the centres of the six neighbouring hexagons.

    Parameters
    ----------
    rmax : float
        The circumradius of each hexagon.

    Returns
    -------
    centers : Array
        The six neighbour centres with shape (6, 2).
    """
    angles = dlu.deg2rad(60 * np.arange(6) + 30)
    return np.sqrt(3.0) * rmax * np.stack((np.cos(angles), np.sin(angles)), axis=-1)


@eqx.filter_jit
def segmented_hex_cens(nrings: int, rmax: float, gap: float = 0.0) -> Array:
    """Hex-segment centres including the central segment.

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

        end = np.roll(outer, -1, axis=0)
        fraction = np.linspace(0.0, 1.0, i + 1)[1:-1]
        mids = outer[:, None] + fraction[None, :, None] * (end - outer)[:, None]
        shaped = mids.reshape([6 * (i - 1), 2])
        cens.append(shaped)

    return np.concatenate(cens)


def non_redundant_support(apertures: Array) -> Array:
    """Returns a non-redundant support mask for overlapping sub-apertures.

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
    array_diameter: float | None = None,
    spider_width: float | None = None,
    spider_angles: list | tuple | Array | None = None,
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
    return_support: bool = False,
) -> Array | tuple[Array, ...]:
    """Builds a static circular aperture.

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
    array_diameter : float | None = None
        Optional diameter of the array. If None, defaults to `diameter`.
    spider_width : float | None = None
        Optional spider vane width.
    spider_angles : list | tuple | Array | None = None
        Angles of spider vanes in degrees.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the Zernike basis diameter.
    return_support : bool = False
        Whether to return the aperture support mask along with the transmission and
        basis. Only relevant if Zernike basis is requested.

    Returns
    -------
    transmission
        Shape (npixels, npixels) if ``zernike_nolls is None``.
    transmission, basis
        If Zernikes are requested, basis has shape
        ``(nterms, npixels, npixels)``.
    transmission, basis, support
        If Zernikes are requested and ``return_support=True``, support has shape
        ``(npixels, npixels)``.
    """
    if (spider_width is None) != (spider_angles is None):
        raise ValueError(
            "`spider_width` and `spider_angles` must both be provided or both be None."
        )

    # Get the oversampled primary aperture
    coord_diam = diameter if array_diameter is None else array_diameter
    coords = dlu.pixel_coords(npixels * oversample, diameter=coord_diam)
    layers = [dlu.circle(coords, diameter)]

    # Add the secondary if requested
    if secondary_diameter is not None and secondary_diameter > 0:
        layers.append(dlu.circle(coords, secondary_diameter, invert=True))

    # Add the spiders if requested
    if spider_width is not None:
        layers += [dlu.spider(coords, spider_width, spider_angles)]

    # Get the combined transmission of the layers
    transmission = dlu.combine(layers, oversample)

    # Return the transmission if no Zernike basis is requested
    if zernike_nolls is None:
        return transmission

    # Get the non-oversampled Zernike basis for the primary aperture
    coords = dlu.pixel_coords(npixels, diameter=coord_diam)
    z_diam = diameter * (1.0 + zernike_oversize)
    basis = dlu.zernike_basis(zernike_nolls, coords, z_diam)

    # Mask the basis by the primary aperture (not including secondary or spiders)
    support = dlu.downsample(layers[0], oversample) > 0
    basis = basis * support[None, ...]

    # Return the transmission, basis, and support if requested
    if return_support:
        return transmission, basis, support

    # Return the transmission and basis
    return transmission, basis


def segmented_aperture(
    npixels: int,
    diameter: float,
    nrings: int,
    segment_diameter: float,
    gap: float = 0.0,
    oversample: int = 5,
    nrings_excluded: int = 1,
    secondary_diameter: float | None = None,
    spider_width: float | None = None,
    spider_angles: list | tuple | Array | None = None,
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
    return_support: bool = False,
) -> Array | tuple[Array, ...]:
    """Builds a static segmented hexagonal aperture.

    Parameters
    ----------
    npixels : int
        The output size of the aperture arrays.
    diameter : float
        The full aperture diameter.
    nrings : int
        The number of hexagonal rings including the center segment.
    segment_diameter : float
        Flat-to-flat diameter of each segment.
    gap : float = 0.0
        Physical gap between neighbouring segments.
    oversample : int = 5
        The oversampling factor used to build soft pixel edges.
    nrings_excluded : int = 1
        Number of inner rings to remove from the aperture. Set to ``0`` to
        keep all segments; ``1`` removes only the central segment; ``2``
        removes the center plus the first surrounding ring, etc.
    secondary_diameter : float | None = None
        Optional circular secondary obscuration diameter.
    spider_width : float | None = None
        Optional spider vane width.
    spider_angles : list | tuple | Array | None = None
        Angles of spider vanes in degrees.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the segment Zernike basis diameter.
    return_support : bool = False
        Whether to return the aperture support mask along with the transmission and
        basis. Only relevant if Zernike basis is requested.

    Returns
    -------
    transmission
        Shape (npixels, npixels) if ``zernike_nolls is None``.
    transmission, basis
        If Zernikes are requested, basis has shape
        ``(n_segments, nterms, npixels, npixels)``.
    transmission, basis, support
        If Zernikes are requested and ``return_support=True``, support has shape
        ``(n_segments, npixels, npixels)``.

    Notes
    -----
    The per-segment Zernikes are circular Zernikes in each local segment frame,
    clipped only by the binary segment mask.
    """
    if (spider_width is None) != (spider_angles is None):
        raise ValueError(
            "`spider_width` and `spider_angles` must both be provided or both be None."
        )

    # Get the segment centres
    rmax = segment_diameter / 2  # segment_diameter is the circumscribed circle diameter
    cens = segmented_hex_cens(nrings, rmax, gap)

    # Remove inner rings before computing any apertures.
    # Ring k (0-indexed) has 6k segments (ring 0 = 1 center). Cumulative:
    #   1 + 6*(1+2+...+(k-1)) = 1 + 3*k*(k-1)  (centred hexagonal numbers)
    if nrings_excluded > 0:
        n_excluded = 1 + 3 * nrings_excluded * (nrings_excluded - 1)
        cens = cens[n_excluded:]

    # Get the oversampled coordinates
    coords = dlu.pixel_coords(npixels * oversample, diameter=diameter)
    # Get the individual hexagonal segments (only for kept centres)
    hex_fn = lambda c: dlu.reg_polygon(dlu.translate_coords(coords, c), 2 * rmax, 6)
    hexes = vmap(hex_fn)(cens)

    # Build the list of transmission layers
    layers = [hexes.sum(0)]

    # Add the secondary if requested
    if secondary_diameter is not None and secondary_diameter > 0:
        layers.append(dlu.circle(coords, secondary_diameter, invert=True))

    # Add the spiders if requested
    if spider_width is not None:
        layers.append(dlu.spider(coords, spider_width, spider_angles))

    # Get the combined transmission of the layers
    transmission = dlu.combine(layers, oversample)

    # Return the transmission if no Zernike basis is requested
    if zernike_nolls is None:
        return transmission

    # Get the non-oversampled Zernike basis for each segment
    coords = dlu.pixel_coords(npixels, diameter=diameter)
    # Get the zernike generation function
    z_diam = segment_diameter * (1.0 + zernike_oversize)
    z_fn = lambda c: dlu.zernike_basis(
        zernike_nolls, dlu.translate_coords(coords, c), z_diam
    )

    # Get the downsampled segment masks and supports
    hexes = dlu.downsample(hexes, oversample)
    seg_support = non_redundant_support(hexes)

    # Calculate the basis for each segment and mask by the segment shape
    basis = vmap(z_fn)(cens) * seg_support[:, None]

    # Return the transmission, basis, and support if requested
    if return_support:
        return transmission, basis, seg_support

    # Return the transmission and basis
    return transmission, basis


def sparse_aperture(
    npixels: int,
    diameter: float,
    centers: list | tuple | Array,
    hole_diameter: float,
    shape: str = "circle",
    oversample: int = 5,
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
    return_support: bool = False,
) -> Array | tuple[Array, ...]:
    """Builds a static sparse aperture from explicit sub-aperture centres.

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
    return_support : bool = False
        Whether to return the aperture support mask along with the transmission and
        basis. Only relevant if Zernike basis is requested.

    Returns
    -------
    transmission
        Shape (npixels, npixels) if ``zernike_nolls is None``.
    transmission, basis
        If Zernikes are requested, basis has shape
        ``(n_ap, nterms, npixels, npixels)``.
    transmission, basis, support
        If Zernikes are requested and ``return_support=True``, support has shape
        ``(n_ap, npixels, npixels)``.
    """
    if shape not in {"circle", "hex"}:
        raise ValueError("`shape` must be either 'circle' or 'hex'.")

    # Get the oversampled coordinates
    coords = dlu.pixel_coords(npixels * oversample, diameter=diameter)
    # Pick the sub-aperture shape function
    if shape == "circle":
        aperture_fn = lambda c: dlu.circle(
            dlu.translate_coords(coords, c), hole_diameter
        )
    else:
        rmax = hole_diameter / np.sqrt(3.0)
        aperture_fn = lambda c: dlu.reg_polygon(
            dlu.translate_coords(coords, c), 2 * rmax, 6
        )

    # Get the individual sub-apertures
    centers = np.array(centers, float)
    apers = vmap(aperture_fn)(centers)

    # Get the combined transmission of the layers
    transmission = dlu.combine(apers, oversample, use_sum=True)

    # Return the transmission if no Zernike basis is requested
    if zernike_nolls is None:
        return transmission

    # Get the non-oversampled Zernike basis for each sub-aperture
    coords = dlu.pixel_coords(npixels, diameter=diameter)
    # Get the zernike basis function
    z_diam = hole_diameter * (1.0 + zernike_oversize)
    if shape == "hex":
        z_diam *= np.sqrt(3.0)
    z_fn = lambda c: dlu.zernike_basis(
        zernike_nolls, dlu.translate_coords(coords, c), z_diam
    )

    # Get the downsampled sub aperture support
    supp = dlu.downsample(apers, oversample) > 0

    # Calculate the basis for each sub-aperture and mask by the sub-aperture shape
    basis = vmap(z_fn)(centers) * supp[:, None]

    # Return the transmission, basis, and support if requested
    if return_support:
        return transmission, basis, supp

    # Return the transmission and basis
    return transmission, basis


def hst_like(
    npixels: int,
    diameter: float = 2.4,
    oversample: int = 5,
    secondary_diameter: float = 0.305,
    array_diameter: float | None = None,
    spider_width: float | None = 0.038,
    spider_angles: list | tuple = (0, 90, 180, 270),
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
    return_support: bool = False,
) -> Array | tuple[Array, ...]:
    """Builds an HST-like circular aperture model.

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
    array_diameter : float | None = None
        Optional diameter of the array. If None, defaults to `diameter`.
    spider_width : float | None = 0.038
        Width of the spider vanes.
    spider_angles : list | tuple = (0, 90, 180, 270)
        Spider vane angles in degrees.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the Zernike basis diameter.
    return_support : bool = False
        Whether to return the aperture support mask along with the transmission and
        basis. Only relevant if Zernike basis is requested.

    Returns
    -------
    transmission : Array
        The HST-like aperture transmission.
    transmission, basis : tuple[Array, Array]
        Returned when ``zernike_nolls`` is provided.
    transmission, basis, support : tuple[Array, Array, Array]
        Returned when ``zernike_nolls`` is provided and ``return_support=True``.
    """
    return circular_aperture(
        npixels=npixels,
        diameter=diameter,
        oversample=oversample,
        secondary_diameter=secondary_diameter,
        array_diameter=array_diameter,
        spider_width=spider_width,
        spider_angles=spider_angles,
        zernike_nolls=zernike_nolls,
        zernike_oversize=zernike_oversize,
        return_support=return_support,
    )


def jwst_like(
    npixels: int,
    diameter: float = 6.6,
    nrings: int = 3,
    segment_diameter: float = 1.524,  # 2 * 1.32 / sqrt(3), from flat-to-flat 1.32m
    gap: float = 0.007,
    oversample: int = 5,
    nrings_excluded: int = 1,
    secondary_diameter: float | None = None,
    spider_width: float | None = 0.1,
    spider_angles: list | tuple = (30, 180, 330),
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
    return_support: bool = False,
) -> Array | tuple[Array, ...]:
    """Builds a JWST-like segmented aperture model.

    Parameters
    ----------
    npixels : int
        The output size of the aperture arrays.
    diameter : float = 6.6
        The full primary diameter.
    nrings : int = 3
        The number of hexagonal rings including the central segment.
    segment_diameter : float = 1.524
        Circumscribed circle diameter of each segment.
    gap : float = 0.007
        Segment spacing.
    oversample : int = 5
        The oversampling factor used to build soft pixel edges.
    nrings_excluded : int = 1
        Number of inner rings to remove. Defaults to ``1`` (removes the
        central segment).
    secondary_diameter : float | None = None
        Optional circular secondary obscuration diameter.
    spider_width : float | None = 0.1
        Width of the spider vanes.
    spider_angles : list | tuple = (30, 180, 330)
        Spider vane angles in degrees.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the Zernike basis diameter.
    return_support : bool = False
        Whether to return the aperture support mask along with the transmission and
        basis. Only relevant if Zernike basis is requested.

    Returns
    -------
    transmission : Array
        The JWST-like aperture transmission.
    transmission, basis : tuple[Array, Array]
        Returned when ``zernike_nolls`` is provided.
    transmission, basis, support : tuple[Array, Array, Array]
        Returned when ``zernike_nolls`` is provided and ``return_support=True``.
    """
    return segmented_aperture(
        npixels=npixels,
        diameter=diameter,
        nrings=nrings,
        segment_diameter=segment_diameter,
        gap=gap,
        oversample=oversample,
        nrings_excluded=nrings_excluded,
        secondary_diameter=secondary_diameter,
        spider_width=spider_width,
        spider_angles=spider_angles,
        zernike_nolls=zernike_nolls,
        zernike_oversize=zernike_oversize,
        return_support=return_support,
    )


def euclid_like(
    npixels: int,
    diameter: float = 1.21,
    oversample: int = 5,
    secondary_diameter: float = 0.395,
    array_diameter: float | None = None,
    spider_width: float = 0.012,
    spider_angles: list | tuple = (0, 120, 240),
    zernike_nolls: list | tuple | Array | None = None,
    zernike_oversize: float = 0.01,
    return_support: bool = False,
) -> Array | tuple[Array, ...]:
    """Builds a Euclid-like circular aperture model.

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
    array_diameter : float | None = None
        Optional diameter of the array. If None, defaults to `diameter`.
    spider_width : float = 0.012
        Width of the spider vanes.
    spider_angles : list | tuple = (0, 120, 240)
        Spider vane angles in degrees.
    zernike_nolls : list | tuple | Array | None = None
        Optional Noll indices for Zernike basis generation.
    zernike_oversize : float = 0.01
        Fractional oversize of the Zernike basis diameter.
    return_support : bool = False
        Whether to return the aperture support mask along with the transmission and
        basis. Only relevant if Zernike basis is requested.

    Returns
    -------
    aperture : Array
        The Euclid-like aperture transmission.
    aperture, basis : tuple[Array, Array]
        Returned when ``zernike_nolls`` is provided.
    aperture, basis, support : tuple[Array, Array, Array]
        Returned when ``zernike_nolls`` is provided and ``return_support=True``.

    Notes
    -----
    This is an approximation that captures the main Euclid aperture features.
    """
    # Get the oversampled aperture without spiders or zernikes
    aperture = circular_aperture(
        npixels=npixels * oversample,
        diameter=diameter,
        secondary_diameter=secondary_diameter,
        array_diameter=array_diameter,
    )

    # Get the coordinates for the spiders
    ap_coord_diam = diameter if array_diameter is None else array_diameter
    ap_coords = dlu.pixel_coords(npixels * oversample, diameter=ap_coord_diam)

    # Get the generation functions
    spider_shift = np.array([secondary_diameter / 2 - spider_width / 2, diameter / 2])
    rot_fn = eqx.filter_jit(
        lambda angle: dlu.rotate_coords(ap_coords, dlu.deg2rad(angle + 30))
    )
    shift_fn = eqx.filter_jit(
        lambda angle: dlu.translate_coords(rot_fn(angle), spider_shift)
    )
    rect_fn = eqx.filter_jit(
        lambda c: dlu.rectangle(c, spider_width, diameter, invert=True)
    )

    # Get the spider vanes
    spiders = vmap(lambda angle: rect_fn(shift_fn(angle)))(np.asarray(spider_angles))

    # Combine the aperture and spiders, and downsample back to npixels
    aperture = dlu.combine([aperture] + list(spiders), oversample)

    if zernike_nolls is None:
        return aperture

    # Get the basis
    _, basis, support = circular_aperture(
        npixels=npixels,
        diameter=diameter,
        oversample=oversample,
        secondary_diameter=secondary_diameter,
        array_diameter=array_diameter,
        zernike_nolls=zernike_nolls,
        zernike_oversize=zernike_oversize,
        return_support=True,
    )

    # Mask the basis by the aperture support
    if return_support:
        return aperture, basis, support

    # Return the aperture and basis
    return aperture, basis
