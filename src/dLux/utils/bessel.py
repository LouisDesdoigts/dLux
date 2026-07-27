"""
Yoinked from JAX-GalSim
"""

import jax
import jax.numpy as np

# =====================================================================
# Modified Bessel K_v(x)
#
# The functions below (_sqrt1px2, _evaluate_temme_coeffs,
# _temme_series_kve, _continued_fraction_kve, _olver_kve, _temme_kve,
# _kve_core) and the _ASYMPTOTIC_OLVER_COEFFICIENTS constant are
# derived from TensorFlow Probability's bessel.py:
#   https://github.com/tensorflow/probability
#
# Original copyright and license:
#   Copyright 2020 The TensorFlow Probability Authors.
#   Licensed under the Apache License, Version 2.0.
#
# Modifications from the original:
#   - Ported from TensorFlow/TFP APIs to pure JAX (jax.numpy, jax.lax)
#   - Removed I_v (bessel_ive) computation; only K_v is computed
#   - Removed log-space output option
#   - Removed negative-v correction for I_v
#   - Simplified to scalar-only core (vectorization via jax.vmap)
# =====================================================================

# Olver expansion polynomial coefficients (10 terms, up to 31 coefficients each)
# fmt: off
_ASYMPTOTIC_OLVER_COEFFICIENTS = [
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     -0.20833333333333334, 0., 0.125, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.3342013888888889, 0.,
     -0.40104166666666669, 0., 0.0703125, 0., 0.0],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., -1.0258125964506173, 0., 1.8464626736111112,
     0., -0.89121093750000002, 0., 0.0732421875, 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 4.6695844234262474, 0., -11.207002616222995, 0.,
     8.78912353515625, 0., -2.3640869140624998, 0., 0.112152099609375,
     0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     -28.212072558200244, 0., 84.636217674600744, 0., -91.818241543240035,
     0., 42.534998745388457, 0., -7.3687943594796312, 0., 0.22710800170898438,
     0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 212.5701300392171, 0.,
     -765.25246814118157, 0., 1059.9904525279999, 0., -699.57962737613275,
     0., 218.19051174421159, 0., -26.491430486951554, 0., 0.57250142097473145,
     0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., -1919.4576623184068, 0.,
     8061.7221817373083, 0., -13586.550006434136, 0., 11655.393336864536,
     0., -5305.6469786134048, 0., 1200.9029132163525, 0.,
     -108.09091978839464, 0., 1.7277275025844574, 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 20204.291330966149, 0., -96980.598388637503, 0.,
     192547.0012325315, 0., -203400.17728041555, 0., 122200.46498301747,
     0., -41192.654968897557, 0., 7109.5143024893641, 0.,
     -493.915304773088, 0., 6.074042001273483, 0., 0., 0., 0., 0.,
     0., 0., 0.],
    [0., 0., 0., -242919.18790055133, 0., 1311763.6146629769, 0.,
     -2998015.9185381061, 0., 3763271.2976564039, 0., -2813563.2265865342, 0.,
     1268365.2733216248, 0., -331645.17248456361, 0., 45218.768981362737, 0.,
     -2499.8304818112092, 0., 24.380529699556064, 0., 0., 0., 0., 0.,
     0., 0., 0., 0.0],
    [3284469.8530720375, 0., -19706819.11843222, 0., 50952602.492664628,
     0., -74105148.211532637, 0., 66344512.274729028, 0., -37567176.660763353,
     0., 13288767.166421819, 0., -2785618.1280864552, 0., 308186.40461266245,
     0., -13886.089753717039, 0., 110.01714026924674, 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.]
]
# fmt: on


def _sqrt1px2(x):
    """Numerically stable computation of sqrt(1 + x^2)."""
    eps = np.finfo(x.dtype).eps
    return np.where(
        np.abs(x) * np.sqrt(eps) <= 1.0,
        np.exp(0.5 * np.log1p(x * x)),
        np.abs(x),
    )


def _evaluate_temme_coeffs(v):
    """Numerically stable computation of gamma-related coefficients for Temme's method.

    Computes:
      coeff1 = (1/Gamma(1-v) - 1/Gamma(1+v)) / (2v)
      coeff2 = (1/Gamma(1-v) + 1/Gamma(1+v)) / 2
      gamma1pv = 1/Gamma(1+v)
      gamma1mv = 1/Gamma(1-v)

    Uses Chebyshev expansions for numerical stability (avoids catastrophic cancellation).
    """
    coeff1_coeffs = [
        -1.142022680371168e0,
        6.5165112670737e-3,
        3.087090173086e-4,
        -3.4706269649e-6,
        6.9437664e-9,
        3.67795e-11,
        -1.356e-13,
    ]
    coeff2_coeffs = [
        1.843740587300905e0,
        -7.68528408447867e-2,
        1.2719271366546e-3,
        -4.9717367042e-6,
        -3.31261198e-8,
        2.423096e-10,
        -1.702e-13,
        -1.49e-15,
    ]
    w = 8.0 * v * v - 1.0
    y = 2.0 * w

    # Clenshaw's recurrence for coeff1
    prev = 0.0
    current = 0.0
    for i in reversed(range(1, len(coeff1_coeffs))):
        temp = current
        current = y * current - prev + coeff1_coeffs[i]
        prev = temp
    coeff1 = w * current - prev + 0.5 * coeff1_coeffs[0]

    # Clenshaw's recurrence for coeff2
    prev = 0.0
    current = 0.0
    for i in reversed(range(1, len(coeff2_coeffs))):
        temp = current
        current = y * current - prev + coeff2_coeffs[i]
        prev = temp
    coeff2 = w * current - prev + 0.5 * coeff2_coeffs[0]

    gamma1pv = coeff2 - v * coeff1
    gamma1mv = coeff2 + v * coeff1
    return coeff1, coeff2, gamma1pv, gamma1mv


def _temme_series_kve(v, z):
    """Compute Kve(v, z) and Kve(v+1, z) via Temme power series.

    Assumes |v| < 0.5 and |z| <= 2 for fast convergence.
    Returns exponentially scaled values: Kv(v,z)*exp(z).
    """
    tol = np.finfo(z.dtype).eps

    coeff1, coeff2, gamma1pv_inv, gamma1mv_inv = _evaluate_temme_coeffs(v)

    z_sq = z * z
    logzo2 = np.log(z / 2.0)
    mu = -v * logzo2
    sinc_v = np.sinc(v)
    mu_msk = np.where(mu == 0.0, 1.0, mu)
    sinhc_mu = np.where(mu == 0.0, 1.0, np.sinh(mu_msk) / mu_msk)

    initial_f = (coeff1 * np.cosh(mu) + coeff2 * (-logzo2) * sinhc_mu) / sinc_v
    initial_p = 0.5 * np.exp(mu) / gamma1pv_inv
    initial_q = 0.5 * np.exp(-mu) / gamma1mv_inv

    max_iterations = 1000

    def body_fn(carry):
        should_stop, index, f, p, q, coeff, kv_sum, kvp1_sum = carry
        f = np.where(
            should_stop,
            f,
            (index * f + p + q) / (index * index - v * v),
        )
        p = np.where(should_stop, p, p / (index - v))
        q = np.where(should_stop, q, q / (index + v))
        h = p - index * f
        coeff = np.where(should_stop, coeff, coeff * z_sq / (4.0 * index))
        kv_sum = np.where(should_stop, kv_sum, kv_sum + coeff * f)
        kvp1_sum = np.where(should_stop, kvp1_sum, kvp1_sum + coeff * h)
        index = index + 1.0
        should_stop = (np.abs(coeff * f) < np.abs(kv_sum) * tol) | (
            index > max_iterations
        )
        return (should_stop, index, f, p, q, coeff, kv_sum, kvp1_sum)

    def cond_fn(carry):
        should_stop = carry[0]
        return ~should_stop

    init = (
        np.array(False),
        1.0,
        initial_f,
        initial_p,
        initial_q,
        1.0,
        initial_f,
        initial_p,
    )
    _, _, _, _, _, _, kv_sum, kvp1_sum = jax.lax.while_loop(cond_fn, body_fn, init)

    # Convert to exponentially scaled: kve = kv * exp(z)
    kve = kv_sum * np.exp(z)
    kvep1 = 2.0 * kvp1_sum * np.exp(z) / z

    return kve, kvep1


def _continued_fraction_kve(v, z):
    """Compute Kve(v, z) and Kve(v+1, z) via Steed's continued fraction.

    Assumes |v| < 0.5 and |z| > 2.
    Returns exponentially scaled values: Kv(v,z)*exp(z).
    """
    tol = np.finfo(z.dtype).eps
    max_iterations = 1000

    initial_numerator = v * v - 0.25
    initial_denominator = 2.0 * (z + 1.0)
    initial_ratio = 1.0 / initial_denominator
    initial_seq = -initial_numerator

    def steeds_body(carry):
        (
            should_stop,
            index,
            partial_numerator,
            partial_denominator,
            denominator_ratio,
            convergent_difference,
            hypergeometric_ratio,
            k_0,
            k_1,
            c,
            q,
            hypergeometric_sum,
        ) = carry

        partial_numerator = partial_numerator - 2.0 * (index - 1.0)
        c = np.where(should_stop, c, -c * partial_numerator / index)
        next_k = (k_0 - partial_denominator * k_1) / partial_numerator
        k_0 = np.where(should_stop, k_0, k_1)
        k_1 = np.where(should_stop, k_1, next_k)
        q = np.where(should_stop, q, q + c * next_k)
        partial_denominator = partial_denominator + 2.0
        denominator_ratio = 1.0 / (
            partial_denominator + partial_numerator * denominator_ratio
        )
        convergent_difference = np.where(
            should_stop,
            convergent_difference,
            convergent_difference * (partial_denominator * denominator_ratio - 1.0),
        )
        hypergeometric_ratio = np.where(
            should_stop,
            hypergeometric_ratio,
            hypergeometric_ratio + convergent_difference,
        )
        hypergeometric_sum = np.where(
            should_stop,
            hypergeometric_sum,
            hypergeometric_sum + q * convergent_difference,
        )
        index = index + 1.0
        should_stop = (
            np.abs(q * convergent_difference) < np.abs(hypergeometric_sum) * tol
        ) | (index > max_iterations)
        return (
            should_stop,
            index,
            partial_numerator,
            partial_denominator,
            denominator_ratio,
            convergent_difference,
            hypergeometric_ratio,
            k_0,
            k_1,
            c,
            q,
            hypergeometric_sum,
        )

    def cond_fn(carry):
        return ~carry[0]

    init = (
        np.array(False),
        2.0,
        initial_numerator,
        initial_denominator,
        initial_ratio,
        initial_ratio,
        initial_ratio,
        0.0,
        1.0,
        initial_seq,
        initial_seq,
        1.0 - initial_numerator * initial_ratio,
    )
    result = jax.lax.while_loop(cond_fn, steeds_body, init)
    hypergeometric_ratio = result[6]
    hypergeometric_sum = result[11]

    # log(kve) = 0.5*log(pi/(2z)) - log(hypergeometric_sum)
    log_kve = 0.5 * np.log(np.pi / (2.0 * z)) - np.log(hypergeometric_sum)
    log_kvp1e = (
        log_kve
        + np.log1p(2.0 * (v + z + initial_numerator * hypergeometric_ratio))
        - np.log(z)
        - np.log(2.0)
    )
    return np.exp(log_kve), np.exp(log_kvp1e)


def _olver_kve(v, z):
    """Compute Kve(v, z) using Olver's uniform asymptotic expansion.

    Valid for |v| >= 50. Returns exponentially scaled value: Kv(v,z)*exp(z).
    """
    v_abs = np.abs(v)
    w = z / v_abs
    t = 1.0 / _sqrt1px2(w)

    divisor = v_abs
    kve_sum = 1.0

    # Evaluate the Olver polynomial terms using Horner's method
    for i in range(len(_ASYMPTOTIC_OLVER_COEFFICIENTS)):
        coeff = 0.0
        for c in _ASYMPTOTIC_OLVER_COEFFICIENTS[i]:
            coeff = coeff * t + c
        term = coeff / divisor
        # For K_v, signs alternate: (-1)^i
        kve_sum = kve_sum + (term if i % 2 == 1 else -term)
        divisor = divisor * v_abs

    # log(kve) = 0.5*log(pi*t/(2*v_abs)) - v_abs*shared_prefactor
    shared_prefactor = 1.0 / (_sqrt1px2(w) + w) + np.log(w) - np.log1p(1.0 / t)
    log_k_prefactor = 0.5 * np.log(np.pi * t / (2.0 * v_abs)) - v_abs * shared_prefactor

    log_kve = log_k_prefactor + np.log(kve_sum)
    return np.exp(log_kve)


def _temme_kve(v, x):
    """Compute Kve(v, x) using Temme's method for |v| < 50.

    Reduces to fractional order |u| <= 0.5, computes Kve(u, x) and Kve(u+1, x),
    then uses forward recurrence to reach order v.
    Returns exponentially scaled value: Kv(v,x)*exp(x).
    """
    v = np.abs(v)
    n = np.round(v)
    u = v - n

    # Branchless: compute both methods with safe inputs, select with np.where
    small_x = np.where(x <= 2.0, x, 0.1)
    large_x = np.where(x > 2.0, x, 1000.0)

    temme_kue, temme_kuep1 = _temme_series_kve(u, small_x)
    cf_kue, cf_kuep1 = _continued_fraction_kve(u, large_x)

    kue = np.where(x <= 2.0, temme_kue, cf_kue)
    kuep1 = np.where(x <= 2.0, temme_kuep1, cf_kuep1)

    # Forward recurrence: K_{v+1}(z) = (2v/z)*K_v(z) + K_{v-1}(z)
    # This recurrence is also satisfied by Kv*exp(z) (the exponentially scaled form).
    def bessel_recurrence(carry):
        index, kve, kvep1 = carry
        next_kvep1 = 2.0 * (u + index) * kvep1 / x + kve
        kve = np.where(index > n, kve, kvep1)
        kvep1 = np.where(index > n, kvep1, next_kvep1)
        return (index + 1.0, kve, kvep1)

    def recurrence_cond(carry):
        index = carry[0]
        return index <= n

    _, kve, _ = jax.lax.while_loop(
        recurrence_cond, bessel_recurrence, (1.0, kue, kuep1)
    )
    return kve


def _kve_core(nu, x):
    """Core dispatcher for Kve(nu, x) = Kv(nu, x) * exp(x).

    Branchless: computes both Olver and Temme with safe dummy inputs,
    selects based on |nu| >= 50.
    """
    nu = np.abs(nu)

    # Safe inputs: avoid invalid regions for each method
    small_nu = np.where(nu < 50.0, nu, 0.1)
    large_nu = np.where(nu >= 50.0, nu, 1000.0)

    olver_result = _olver_kve(large_nu, x)
    temme_result = _temme_kve(small_nu, x)

    return np.where(nu >= 50.0, olver_result, temme_result)


def _kv_scalar(nu, x):
    """Scalar implementation of K_v(x) using TFP-ported Temme + Olver algorithms."""
    nu = 1.0 * nu
    x = 1.0 * x
    nu = np.abs(nu)  # K_{-v} = K_v

    # Compute via exponentially scaled form for numerical stability
    # Use a safe x for the core computation (avoid x=0 which causes issues)
    safe_x = np.where(x > 0.0, x, 1.0)
    kve = _kve_core(nu, safe_x)
    result = kve * np.exp(-safe_x)

    # Edge cases
    result = np.where(x == 0.0, np.inf, result)
    result = np.where(x < 0.0, np.nan, result)
    return result


@jax.jit
def _kv_impl(nu, x):
    """Modified Bessel function of the second kind K_v(x) - internal implementation.

    Uses TFP-ported Temme + Olver algorithms for all orders.
    Handles both scalar and array inputs via jax.vmap.
    """
    nu = np.asarray(1.0 * nu)
    x = np.asarray(1.0 * x)
    out_shape = np.broadcast_shapes(np.shape(nu), np.shape(x))
    if out_shape == ():
        return _kv_scalar(nu, x)
    nu_bc, x_bc = np.broadcast_arrays(nu, x)
    flat_nu = nu_bc.ravel()
    flat_x = x_bc.ravel()
    return jax.vmap(_kv_scalar)(flat_nu, flat_x).reshape(out_shape)


@jax.custom_vjp
def kv(nu, x):
    """Modified Bessel function of the second kind K_v(x) with custom gradients.

    Uses TFP-ported Temme + Olver algorithms. Custom gradients via:
        dK_v/dx = -1/2 * (K_{v-1}(x) + K_{v+1}(x))

    Gradient w.r.t. v is not supported (returns zero).
    """
    return _kv_impl(nu, x)


def _kv_fwd(nu, x):
    kv_val = _kv_impl(nu, x)
    kv_prev = _kv_impl(nu - 1.0, x)
    kv_next = _kv_impl(nu + 1.0, x)
    return kv_val, (nu, x, kv_prev, kv_next)


def _kv_bwd(residuals, g):
    nu, x, kv_prev, kv_next = residuals
    grad_x = -0.5 * (kv_prev + kv_next) * g
    grad_nu = np.zeros_like(nu)
    return (grad_nu, grad_x)


kv.defvjp(_kv_fwd, _kv_bwd)


kv(5 / 6, 2 * np.pi * 1.0)  # Example usage
