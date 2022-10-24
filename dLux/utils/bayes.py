import jax.numpy as np
from jax import hessian, grad
from jax.scipy.stats import poisson
# from typing import Callable

__all__ = ["poiss_logpri", "chi2_logpri", "calc_cov", "calc_ent"]


Array = np.ndarray


# # NOTE: I think all of these should be likelihoods, not priors

# def poissonian_prior(X         : Array,
#                      data      : Array,
#                      model     : object,
#                      model_fn  : str,
#                      update_fn : Callable) -> Array:
#     """
#     Calculates the poissonion prior after updating the model with the values
#     store in the X array.

#     Parameters
#     ----------
#     X : Array
#         The array of values to update the model with.
#     data : Array
#         The data to calulcate the prior with respect to.
#     model : object
#         The model used to model the data.
#     model_fn : str
#         The name of the model function to use.
#     update_fn : Callable
#         The function used to update the model before modelling. This should
#         take in the X array as the first argument and the 

#     Returns
#     -------
#     prior : Array
#         The poissonian prior of the updated model with respect to the data.
#     """
#     model = update_fn(X, model)
#     psf = getattr(model, "model_fn")()
#     return np.sum(poisson.pmf(data, psf))

def poiss_logpri(X : Array,
                data : Array,
                model,
                update_fn):
    """

    """
    model = update_fn(X, model)
    psf = model.model()
    return np.sum(poisson.logpmf(data, psf))

def chi2_logpri(X, data, model, update_fn, noise=1):
    """

    """
    model = update_fn(X, model)
    psf = model.model()
    return np.log(np.sum(np.square((psf-data)/noise)))

def calc_cov(loglike_fn, *args):
    """
    Calcuates the covariance matrix under the Laplace approximation
    """
    matrix = hessian(loglike_fn)(*args)
    return -np.linalg.inv(matrix)

def calc_ent(loglike_fn, *args):
    """
    Calcautes the entropy of the covariance matrix
    """
    cov = calc_cov(loglike_fn, *args)
    sign, logdet = np.linalg.slogdet(cov)
    ent = 0.5 * (np.log(2 * np.pi * np.e) + (sign * logdet))
    return ent