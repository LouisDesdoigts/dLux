"""JIT-compatible infinite von Kármán atmospheric phase screens."""

import dLux.utils as dlu
import equinox as eqx
import interpax as ipx
import jax
import jax.numpy as np

from .optical_layers import OpticalLayer
from jax import Array

__all__ = [
    "AtmosphericLayer",
    "InfiniteAtmosphericLayer",
]


class AtmosphericLayer(OpticalLayer):
    """Base class for thin atmospheric OPD layers.

    It owns the common physical metadata and the current screen state. Concrete
    layers implement their own screen-generation and temporal-evolution models.
    """

    npixels: int = eqx.field(static=True)
    pixel_scale: Array
    Cn_squared: float
    L0: float
    velocity: Array
    height: Array
    coords: Array
    base_opd: Array
    screen: Array
    center: Array
    time: Array
    key: Array

    def __init__(self, npixels, pixel_scale, Cn_squared, L0, velocity=0, height=0):
        if not isinstance(npixels, int) or npixels < 2:
            raise ValueError("npixels must be an integer >= 2.")
        if pixel_scale <= 0 or Cn_squared <= 0 or L0 <= 0:
            raise ValueError("pixel_scale, Cn_squared, and L0 must be positive.")
        velocity = np.asarray(velocity, float)
        if velocity.ndim == 0:
            velocity = np.array([velocity, 0.0])
        if velocity.shape != (2,):
            raise ValueError("velocity must be scalar or have shape (2,).")

        self.npixels = npixels
        self.pixel_scale = np.asarray(pixel_scale, float)
        self.Cn_squared = np.asarray(Cn_squared, float)
        self.L0 = np.asarray(L0, float)
        self.velocity = velocity
        self.height = np.asarray(height, float)
        axis = (np.arange(npixels) - (npixels - 1) / 2) * self.pixel_scale
        self.coords = np.stack((axis, axis))

    def __call__(self, wavefront):
        """Apply the current OPD without advancing the atmospheric state."""
        return wavefront.add_opd(self.opd)


class InfiniteAtmosphericLayer(AtmosphericLayer):
    """Functional Assemat/Wilson/Gendron atmospheric layer.

    Use ``opd, layer = layer.step(dt)``. The returned immutable layer can be
    carried by ``jax.jit`` or ``jax.lax.scan``; calling it applies its current
    OPD to a dLux wavefront without advancing time.
    """

    stencil_length: int = eqx.field(static=True)
    use_interpolation: bool = eqx.field(static=True)
    interpolation_method: str = eqx.field(static=True)
    interpolation_kwargs: tuple = eqx.field(static=True)

    initial_key: Array
    stencils: tuple
    A_matrices: tuple[Array]
    B_matrices: tuple[Array]
    high_filter: Array
    low_basis: Array

    def __init__(
        self,
        npixels,
        pixel_scale,
        Cn_squared,
        L0,
        velocity=0,
        height=0,
        stencil_length=2,
        oversampling=16,
        seed=0,
        use_interpolation=True,
        interpolation_method="linear",
        **kwargs,
    ):
        AtmosphericLayer.__init__(
            self, npixels, pixel_scale, Cn_squared, L0, velocity, height
        )
        if not isinstance(stencil_length, int) or not 1 <= stencil_length < npixels:
            raise ValueError("stencil_length must be in [1, npixels).")
        if not isinstance(oversampling, int) or oversampling < 1:
            raise ValueError("oversampling must be a positive integer.")
        self.stencil_length = stencil_length
        self.use_interpolation = bool(use_interpolation)
        self.interpolation_method = str(interpolation_method)
        kwargs.pop("extrap", None)
        self.interpolation_kwargs = tuple(sorted(kwargs.items()))
        stencil_key, self.initial_key = jax.random.split(jax.random.PRNGKey(seed))
        vk, hk = jax.random.split(stencil_key)
        vertical_stencil, horizontal_stencil = self._stencil(vk, True), self._stencil(
            hk, False
        )
        vertical_ab = self._ab(vertical_stencil, True)
        horizontal_ab = self._ab(horizontal_stencil, False)
        self.stencils = (vertical_stencil, horizontal_stencil)
        self.A_matrices = (vertical_ab[0], horizontal_ab[0])
        self.B_matrices = (vertical_ab[1], horizontal_ab[1])
        self.high_filter, self.low_basis = self._spectral_model(oversampling)
        self.base_opd, self.key = self._initial_screen(self.initial_key)
        self.screen, self.center, self.time = (
            self.base_opd,
            np.zeros(2),
            np.asarray(0.0),
        )

    def _stencil(self, key, vertical):
        n, mask = self.npixels, np.zeros((self.npixels, self.npixels), bool)
        random_points = (
            jax.random.geometric(key, 0.5, (n,)) + self.stencil_length - 1
        ) % n
        if vertical:
            mask = (
                mask.at[: self.stencil_length, :]
                .set(True)
                .at[random_points, np.arange(n)]
                .set(True)
            )
        else:
            mask = (
                mask.at[:, : self.stencil_length]
                .set(True)
                .at[np.arange(n), random_points]
                .set(True)
            )
        return np.nonzero(mask.ravel(), size=int(mask.sum()), fill_value=0)[0]

    def _ab(self, stencil, vertical):
        yy, xx = np.meshgrid(self.coords[1], self.coords[0], indexing="ij")
        if vertical:
            nx, ny = self.coords[0], np.full(
                self.npixels, self.coords[1][0] - self.pixel_scale
            )
        else:
            nx, ny = (
                np.full(self.npixels, self.coords[0][0] - self.pixel_scale),
                self.coords[1],
            )
        px = np.concatenate((xx.ravel()[stencil], nx))
        py = np.concatenate((yy.ravel()[stencil], ny))
        cov = dlu.phase_covariance_von_karman(
            dlu.fried_parameter_from_Cn_squared(1.0, 1.0), self.L0
        )(np.stack((px[:, None] - px, py[:, None] - py)))
        nz = stencil.shape[0]
        zz, xz, zx, xx = cov[:nz, :nz], cov[nz:, :nz], cov[:nz, nz:], cov[nz:, nz:]
        A = np.linalg.solve(zz + 1e-7 * np.eye(nz), xz.T).T
        values, vectors = np.linalg.eigh((xx - A @ zx + (xx - A @ zx).T) / 2)
        return A, vectors * np.sqrt(np.maximum(values, 0.0))[None]

    def _spectral_model(self, oversampling):
        n, df = self.npixels, 1 / (self.npixels * self.pixel_scale)
        f = np.fft.fftfreq(n, self.pixel_scale)
        fy, fx = np.meshgrid(f, f, indexing="ij")
        r0 = dlu.fried_parameter_from_Cn_squared(self.Cn_squared, 1.0)
        high = (
            (
                np.sqrt(
                    dlu.power_spectral_density_von_karman(r0, self.L0)(
                        np.stack((2 * np.pi * fx, 2 * np.pi * fy))
                    )
                    * df**2
                )
                * n**2
            )
            .at[0, 0]
            .set(0.0)
        )
        modes = []
        for level in range(1, max(1, int(np.ceil(np.log2(oversampling)))) + 1):
            spacing = df / 2**level
            modes += [
                (i * spacing, j * spacing, spacing)
                for j in (-1, 0, 1)
                for i in (-1, 0, 1)
                if i or j
            ]
        modes = np.asarray(modes)
        phase = (
            2
            * np.pi
            * (
                modes[:, 0, None, None] * self.coords[0][None, None, :]
                + modes[:, 1, None, None] * self.coords[1][None, :, None]
            )
        )
        low_psd = dlu.power_spectral_density_von_karman(r0, self.L0)(
            np.stack((2 * np.pi * modes[:, 0], 2 * np.pi * modes[:, 1]))
        )
        return high, np.sqrt(low_psd * modes[:, 2] ** 2)[:, None, None] * np.exp(
            1j * phase
        )

    def _initial_screen(self, key):
        high_key, low_key, next_key = jax.random.split(key, 3)
        high = np.fft.ifft2(
            jax.random.normal(high_key, (self.npixels, self.npixels)) * self.high_filter
        ).real
        low = np.real(
            np.sum(
                jax.random.normal(low_key, (self.low_basis.shape[0],))[:, None, None]
                * self.low_basis,
                axis=0,
            )
        )
        return high + low, next_key

    def _extrude(self, screen, key, horizontal, flipped):
        stencil, A, B = (
            (self.stencils[1], self.A_matrices[1], self.B_matrices[1])
            if horizontal
            else (self.stencils[0], self.A_matrices[0], self.B_matrices[0])
        )
        samples = screen.ravel()[self.npixels**2 - 1 - stencil if flipped else stencil]
        key, noise_key = jax.random.split(key)
        new = A @ samples + B @ jax.random.normal(noise_key, (B.shape[1],)) * np.sqrt(
            self.Cn_squared
        )
        if horizontal:
            extended = (
                np.concatenate((screen[:, 1:], new[::-1, None]), 1)
                if flipped
                else np.concatenate((new[:, None], screen[:, :-1]), 1)
            )
        else:
            extended = (
                np.concatenate((screen[1:], new[None, ::-1]), 0)
                if flipped
                else np.concatenate((new[None], screen[:-1]), 0)
            )
        return extended, key

    def _advance(self, screen, key, count, horizontal):
        def body(state):
            i, s, k = state
            s, k = jax.lax.cond(
                count < 0,
                lambda a: self._extrude(*a, horizontal, False),
                lambda a: self._extrude(*a, horizontal, True),
                (s, k),
            )
            return i + 1, s, k

        _, screen, key = jax.lax.while_loop(
            lambda s: s[0] < np.abs(count), body, (np.asarray(0), screen, key)
        )
        return screen, key

    def _sample(self, base, residual):
        if not self.use_interpolation:
            return base
        if self.interpolation_method == "linear" and not self.interpolation_kwargs:
            shift = residual / self.pixel_scale

            def interpolate_axis(values, amount, axis):
                positive = lambda value: np.concatenate(
                    (
                        jax.lax.slice_in_dim(value, 1, self.npixels, axis=axis),
                        jax.lax.slice_in_dim(
                            value, self.npixels - 1, self.npixels, axis=axis
                        ),
                    ),
                    axis=axis,
                )
                negative = lambda value: np.concatenate(
                    (
                        jax.lax.slice_in_dim(value, 0, 1, axis=axis),
                        jax.lax.slice_in_dim(value, 0, self.npixels - 1, axis=axis),
                    ),
                    axis=axis,
                )
                neighbor = jax.lax.cond(amount >= 0, positive, negative, values)
                return values + np.abs(amount) * (neighbor - values)

            shifted = interpolate_axis(base, shift[0], 1)
            return interpolate_axis(shifted, shift[1], 0)

        xq = np.clip(
            self.coords[0][None] + residual[0], self.coords[0][0], self.coords[0][-1]
        )
        yq = np.clip(
            self.coords[1][:, None] + residual[1], self.coords[1][0], self.coords[1][-1]
        )
        xx, yy = np.broadcast_arrays(xq, yq)
        return ipx.interp2d(
            yy.ravel(),
            xx.ravel(),
            self.coords[1],
            self.coords[0],
            base,
            method=self.interpolation_method,
            extrap=False,
            **dict(self.interpolation_kwargs),
        ).reshape(base.shape)

    def step(self, dt):
        dt = np.asarray(dt, self.time.dtype)
        center = self.center + self.velocity * dt
        before = np.round(self.center / self.pixel_scale).astype(int)
        after = np.round(center / self.pixel_scale).astype(int)
        base, key = self._advance(self.base_opd, self.key, after[0] - before[0], True)
        base, key = self._advance(base, key, after[1] - before[1], False)
        screen = self._sample(base, center - after * self.pixel_scale)
        layer = eqx.tree_at(
            lambda o: (o.base_opd, o.screen, o.center, o.time, o.key),
            self,
            (base, screen, center, self.time + dt, key),
        )
        return screen, layer

    def evolve(self, dt, steps):
        """Advance several frames inside one compiled loop.

        This avoids dispatching one accelerator program per frame when only the
        final screen and layer state are needed.
        """
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer.")

        layer = jax.lax.fori_loop(
            0,
            steps,
            lambda _, state: state.step(dt)[1],
            self,
        )
        return layer.screen, layer

    def reset(self, independent=False):
        base, key = self._initial_screen(self.key if independent else self.initial_key)
        layer = eqx.tree_at(
            lambda o: (o.base_opd, o.screen, o.center, o.time, o.key),
            self,
            (base, base, np.zeros_like(self.center), np.zeros_like(self.time), key),
        )
        return base, layer
