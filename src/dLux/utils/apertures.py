import jax.numpy as np
import dLux.utils as dlu
from equinox import filter_jit as jit

__all__ = [
    "circular_aperture",
    "segmented_aperture",
    "sparse_aperture",
]


@jit
def _hex_cens(rmax):
    """Centres of the 6 neighbouring hexagons."""
    r = np.sqrt(3.0) * rmax
    xys = []
    for i in range(6):
        angle_rad = dlu.deg2rad(60 * i + 30)
        xy = r * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        xys.append(xy)
    return np.array(xys)


@jit
def _evenly_spaced_points(point1, point2, n):
    """Return n evenly spaced interior points between two 2D points."""
    x = np.linspace(point1[0], point2[0], n + 2)[1:-1]
    y = np.linspace(point1[1], point2[1], n + 2)[1:-1]
    return np.squeeze(np.column_stack((x, y)))


@jit
def _segmented_hex_cens(nrings, rmax, gap=0.0):
    """
    Hex-segment centres including the central segment.

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


def _non_redundant_support(apertures):
    """Get the non-redundant support of a set of sub-pixel overlapping apertures."""
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
    npix,
    diameter,
    oversample=4,
    secondary_diameter=None,
    spider_width=0.0,
    spider_angles=None,
    zernike_nolls=None,
    zernike_oversize=0.01,
):
    """
    Build a static circular aperture.

    Returns
    -------
    transmission
        Shape (npix, npix) if `zernike_nolls is None`.
    transmission, basis
        If Zernikes are requested, basis has shape (nterms, npix, npix).
    """
    # Get the oversampled primary aperture
    coords = dlu.pixel_coords(npix * oversample, diameter)
    layers = [dlu.circle(coords, diameter / 2)]

    # Add the secondary if requested
    if secondary_diameter is not None and secondary_diameter > 0:
        layers.append(dlu.circle(coords, secondary_diameter / 2, invert=True))

    # Add the spiders if requested
    layers += [dlu.spider(coords, spider_width, spider_angles)]

    # Get the combined transmission of the layers
    transmission = dlu.combine(layers, oversample)

    # Return the transmission if no Zernike basis is requested
    if zernike_nolls is None:
        return transmission

    # Get the non-oversampled Zernike basis for the primary aperture
    coords = dlu.pixel_coords(npix, diameter)
    z_diam = diameter * (1.0 + zernike_oversize)
    basis = dlu.zernike_basis(zernike_nolls, coords, z_diam)

    # Mask the basis by the primary aperture (not including secondary or spiders)
    support = dlu.downsample(layers[0], oversample) > 0
    basis = basis * support[None, ...]

    # Return the transmission and basis
    return transmission, basis


def segmented_aperture(
    npix,
    diameter,
    nrings,
    flat_to_flat,
    gap=0.0,
    oversample=4,
    has_secondary=True,
    spider_width=0.0,
    spider_angles=None,
    zernike_nolls=None,
    zernike_oversize=0.01,
):
    """
    Build a static segmented hexagonal aperture.

    Returns
    -------
    transmission
        Shape (npix, npix) if `zernike_nolls is None`.
    transmission, basis
        If Zernikes are requested, basis has shape
        (n_segments, nterms, npix, npix).

    Notes
    -----
    The per-segment Zernikes are circular Zernikes in each local segment frame,
    clipped only by the binary segment mask.
    """
    # Get the oversampled coordinates
    coords = dlu.pixel_coords(npix * oversample, diameter)
    shift_fn = jit(lambda c: dlu.translate_coords(coords, c))

    # Get the segment centres
    rmax = flat_to_flat / np.sqrt(3.0)
    cens = _segmented_hex_cens(nrings, rmax, gap)

    # Get the individual hexagonal segments
    hex_fn = jit(lambda c: dlu.reg_polygon(shift_fn(c), rmax, 6))

    # Avoid vmap to stop ram blowing up
    hexes = np.array([hex_fn(c) for c in cens])

    # Central segment becomes the secondary obstruction
    if has_secondary:
        hexes = hexes[1:]
        cens = cens[1:]

    # Add the spiders if requested
    spiders = dlu.spider(coords, spider_width, spider_angles)

    # Get the combined transmission of the layers
    transmission = dlu.combine([hexes.sum(0), spiders], oversample)

    # Return the transmission if no Zernike basis is requested
    if zernike_nolls is None:
        return transmission

    # Get the non-oversampled Zernike basis for each segment
    coords = dlu.pixel_coords(npix, diameter)
    shift_fn = jit(lambda c: dlu.translate_coords(coords, c))

    # Get the zernike generation function
    z_diam = np.sqrt(3) * flat_to_flat * (1.0 + zernike_oversize)
    z_fn = lambda c: dlu.zernike_basis(zernike_nolls, shift_fn(c), z_diam)

    # Get the downsampled segment masks and supports
    # seg_support = [dlu.downsample(hex, oversample) > 0 for hex in hexes]
    hexes = np.array([dlu.downsample(hex, oversample) for hex in hexes])
    seg_support = _non_redundant_support(hexes)

    # Calculate the basis for each segment and mask by the segment shape
    basis = [z_fn(cen) * supp[None, ...] for cen, supp in zip(cens, seg_support)]

    # Return the transmission and basis
    return transmission, np.array(basis)


def sparse_aperture(
    npix,
    diameter,
    centers,
    hole_diameter,
    shape="circle",
    oversample=4,
    zernike_nolls=None,
    zernike_oversize=0.01,
):
    """
    Build a static sparse aperture from explicit sub-aperture centres.

    Parameters
    ----------
    shape : {"circle", "hex"}

    Returns
    -------
    transmission
        Shape (npix, npix) if `zernike_nolls is None`.
    transmission, basis
        If Zernikes are requested, basis has shape
        (n_ap, nterms, npix, npix).
    """
    if shape not in {"circle", "hex"}:
        raise ValueError("`shape` must be either 'circle' or 'hex'.")

    # Get the oversampled coordinates
    coords = dlu.pixel_coords(npix * oversample, diameter)
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
    coords = dlu.pixel_coords(npix, diameter)
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
