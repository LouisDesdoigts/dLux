import jax.numpy as np
from jax import hessian, grad
from jax.scipy.stats import poisson

__all__ = ["poiss_logpri", "chi2_logpri", "calc_cov", "calc_ent"]

def poiss_logpri(X, data, model, update_fn):
    """Poissonian log-prior"""
    model = update_fn(X, model)
    psf = model()
    return np.sum(poisson.logpmf(data, psf))

def chi2_logpri(X, data, model, update_fn, noise=1):
    """Chi2 log-prior"""
    model = update_fn(X, model)
    psf = model()
    return np.log(np.sum(np.square((psf-data)/noise)))

def calc_cov(loglike_fn, *args):
    """Calcuates the covariance matrix under the Laplace approximation"""
    matrix = hessian(loglike_fn)(*args)
    return -np.linalg.inv(matrix)

def calc_ent(loglike_fn, *args):
    """Calcautes the entropy of the covariance matrix"""
    cov = calc_cov(loglike_fn, *args)
    sign, logdet = np.linalg.slogdet(cov)
    ent = 0.5 * (np.log(2 * np.pi * np.e) + (sign * logdet))
    return ent