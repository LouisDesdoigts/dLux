"""Estimate point-source positions and fluxes from a sampled image."""

import equinox as eqx
import jax.numpy as np
from jax import grad, lax, vmap
from jax.scipy.signal import fftconvolve

from .fourier import fourier_shift

__all__ = ["estimate_sources"]


@eqx.filter_jit
def _matched_filter(data, psf, weights=None):
    """Return the locally normalised, noise-weighted PSF signal-to-noise map."""
    weights = np.ones_like(data) if weights is None else weights

    # Correlate the weighted data and local PSF power using zero-padded FFTs
    numerator = fftconvolve(weights * data, psf[::-1, ::-1], mode="same")
    denominator = fftconvolve(weights, (psf**2)[::-1, ::-1], mode="same")

    # Rank detections by signal-to-noise where the PSF has weighted support
    eps = np.finfo(data.dtype).eps
    threshold = eps * np.max(denominator)
    scale = np.sqrt(np.maximum(denominator, np.finfo(data.dtype).tiny))
    return np.where(denominator > threshold, numerator / scale, -np.inf)


def _shift_psfs(psf, positions):
    """Construct one shifted PSF image per absolute pixel position."""
    centre = (np.asarray(psf.shape[::-1]) - 1) / 2
    shift_fn = lambda position: fourier_shift(psf, position - centre)
    return vmap(shift_fn)(positions)


def _psf_core_radius(psf, factor=1.5):
    """Estimate an exclusion radius from the PSF half-maximum core."""
    core_area = np.sum(psf >= 0.5 * np.max(psf))
    return factor * np.sqrt(core_area / np.pi)


def _exclude_nearby(detection, positions, detection_separation):
    """Suppress detections near accepted absolute ``(x, y)`` positions."""
    yy, xx = np.mgrid[: detection.shape[0], : detection.shape[1]]
    distance_squared = (xx[..., None] - positions[:, 0]) ** 2
    distance_squared += (yy[..., None] - positions[:, 1]) ** 2
    nearest = np.min(distance_squared, axis=-1)
    return np.where(nearest >= detection_separation**2, detection, -np.inf)


def _fit_flux(data, models, weights=None):
    """Solve all source fluxes jointly by weighted least squares."""
    # Flatten the source models into weighted design-matrix columns
    weights = np.ones_like(data) if weights is None else weights
    design = models.reshape((models.shape[0], -1)).T
    sqrt_weights = np.sqrt(weights.reshape(-1))
    design = design * sqrt_weights[:, None]
    values = data.reshape(-1) * sqrt_weights

    # Solve without forming normal equations or squaring the condition number
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _projected_loss(positions, data, psf, weights=None):
    """Return weighted loss, fitted fluxes, and raw residuals."""
    # Eliminate the linear flux parameters at the current positions
    models = _shift_psfs(psf, positions)
    flux = _fit_flux(data, models, weights)

    # Construct the raw residual and apply weights only to the scalar loss
    residual = data - np.sum(flux[:, None, None] * models, axis=0)
    loss_residual = residual
    if weights is not None:
        loss_residual = residual * np.sqrt(weights)
    return np.sum(loss_residual**2), flux, residual


@eqx.filter_jit
def _fit_catalogue(positions, data, psf, weights=None, radius=1.0, samples=9):
    """Refine each position with one gradient-directed line search."""

    def loss_fn(current_positions):
        return _projected_loss(current_positions, data, psf, weights)[0]

    # Define a symmetric line search for each catalogue position
    offsets = np.linspace(-radius, radius, samples)

    def refine_position(index, current_positions):
        # Use the gradient only to choose the local search direction
        direction = -grad(loss_fn)(current_positions)[index]
        direction /= np.maximum(np.linalg.norm(direction), 1e-12)

        # Evaluate candidates while jointly refitting all source fluxes
        candidate_fn = lambda offset: current_positions.at[index].add(
            offset * direction
        )
        candidates = vmap(candidate_fn)(offsets)
        losses = vmap(loss_fn)(candidates)
        return candidates[np.argmin(losses)]

    # Refine sequentially because each position update changes the joint fit
    positions = lax.fori_loop(0, positions.shape[0], refine_position, positions)

    # Evaluate the final jointly fitted catalogue and residual
    loss, flux, residual = _projected_loss(positions, data, psf, weights)
    return positions, flux, residual, loss


def estimate_sources(
    data,
    psf,
    n_source,
    weights=None,
    background=None,
    detection_separation=None,
    pixel_scale=None,
    radius=1.0,
    samples=9,
):
    """Estimate point-source positions and fluxes for model initialisation.

    Sources are detected greedily by correlating the current residual with the
    supplied PSF. After each detection, all fluxes are solved jointly by weighted
    linear least squares and every position receives one gradient-directed line
    search. The gradient determines only the search direction; candidate positions
    are selected by explicitly refitting the fluxes and comparing residual losses.

    This routine is intended to initialise a later differentiable fit rather than
    provide precision photometry. The outer catalogue-growth loop is executed in
    Python because its shape changes after each detection, while each fixed-shape
    catalogue refinement is JIT compiled.

    Parameters
    ----------
    data : Array
        Two-dimensional image with shape ``(ny, nx)``.
    psf : Array
        Matching two-dimensional PSF image. It is interpreted as centred at
        ``((nx - 1) / 2, (ny - 1) / 2)`` and normalised internally to unit sum.
    n_source : int
        Positive number of sources to estimate.
    weights : Array or None
        Optional non-negative inverse-variance weights matching ``data``. Zero
        weights mask pixels from detection and fitting.
    background : float or None
        Constant background to subtract. If `None`, the image median is used as a
        robust initial estimate and remains fixed during source fitting.
    detection_separation : float or None
        Minimum separation between integer detection seeds in pixels. This does not
        constrain the subsequently refined positions. If `None`, a radius is
        estimated from the half-maximum PSF core.
    pixel_scale : float, Array, or None
        Optional scale applied to the returned centred ``(x, y)`` pixel positions.
        A length-two value may scale the physical axes independently.
    radius : float
        Half-width of each local position line search in pixels.
    samples : int
        Odd number of candidate offsets in each local line search, including zero.

    Returns
    -------
    result : dict[str, Array]
        ``position`` has shape ``(n_source, 2)`` in centred physical ``(x, y)``
        order, ``flux`` contains linear amplitudes for the unit-sum PSF,
        ``background`` is the fixed subtracted value, ``loss`` contains one
        weighted residual sum of squares per catalogue size, and ``residual`` has
        shape ``(n_source, ny, nx)`` after subtracting the background and fitted
        catalogue at each iteration.

    Examples
    --------
    Initialise a single source from a sampled Gaussian PSF:

    >>> y, x = np.mgrid[-8:9, -8:9]
    >>> psf = np.exp(-(x**2 + y**2) / 2)
    >>> psf = psf / psf.sum()
    >>> result = estimate_sources(5 * psf, psf, 1, background=0.0)

    Notes
    -----
    Fourier interpolation assumes a spatially invariant PSF. Padding prevents flux
    wrapping across detector edges, but sources whose PSFs extend beyond the image
    have truncated fitted fluxes. A median background is robust for sparse fields but
    may be biased by broad PSF wings or crowded emission; supply a calibrated value
    when available.
    """
    # Convert and validate the public array contract
    data = np.asarray(data, dtype=float)
    psf = np.asarray(psf, dtype=float)
    if data.ndim != 2 or psf.ndim != 2 or data.shape != psf.shape:
        raise ValueError("data and psf must be matching two-dimensional arrays.")
    if not isinstance(n_source, int) or n_source < 1:
        raise ValueError("n_source must be a positive integer.")
    if not isinstance(samples, int) or samples < 3 or samples % 2 == 0:
        raise ValueError("samples must be an odd integer greater than or equal to 3.")
    if radius <= 0:
        raise ValueError("radius must be greater than zero.")
    radius = np.asarray(radius)

    # Validate and resolve optional weighting and coordinate inputs
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != data.shape or np.any(weights < 0):
            raise ValueError("weights must be non-negative and match data.shape.")
    if detection_separation is not None and detection_separation <= 0:
        raise ValueError("detection_separation must be greater than zero.")
    if pixel_scale is not None:
        pixel_scale = np.squeeze(np.asarray(pixel_scale))
        if pixel_scale.shape not in ((), (2,)):
            raise ValueError("pixel_scale must be scalar or have shape (2,).")

    # Normalise the PSF and resolve detection defaults
    total = psf.sum()
    if total <= 0:
        raise ValueError("psf must have a positive sum.")
    template = psf / total
    if detection_separation is None:
        detection_separation = _psf_core_radius(template)
    background = np.median(data) if background is None else np.asarray(background)
    if background.shape != ():
        raise ValueError("background must be scalar or None.")

    # Initialise the background-subtracted catalogue fit
    data = data - background
    residual = data
    positions = []
    flux = np.zeros(0)
    loss_history = []
    residual_history = []

    # Grow and jointly refit the catalogue one source at a time
    for _ in range(n_source):
        detection = _matched_filter(residual, template, weights)
        if positions:
            detection = _exclude_nearby(
                detection,
                np.stack(positions),
                detection_separation,
            )
        if not np.any(np.isfinite(detection)):
            raise ValueError(
                "No finite detection candidates remain; reduce detection_separation "
                "or check the weights and input arrays."
            )
        iy, ix = np.unravel_index(np.argmax(detection), detection.shape)
        positions.append(np.asarray((ix, iy), dtype=float))

        positions, flux, residual, loss = _fit_catalogue(
            np.stack(positions),
            data,
            template,
            weights,
            radius,
            samples,
        )
        positions = list(positions)
        loss_history.append(loss)
        residual_history.append(residual)

    # Convert absolute pixels to centred output coordinates
    positions = np.stack(positions) - (np.asarray(data.shape[::-1]) - 1) / 2
    if pixel_scale is not None:
        positions = positions * pixel_scale

    return {
        "position": positions,
        "flux": flux,
        "background": background,
        "loss": np.stack(loss_history),
        "residual": np.stack(residual_history),
    }
