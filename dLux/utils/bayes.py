import jax.numpy as np
from jax import hessian, grad
from jax.scipy.stats import poisson
from typing import Callable

__all__ = ["poisson_likelihood", "poisson_log_likelihood",
           "chi2_likelihood", "chi2_log_likelihood",
           "calculate_covariance", "calculate_entropy"]


Array = np.ndarray


def poisson_likelihood(X         : Array,
                       data      : Array,
                       model     : object,
                       update_fn : Callable,
                       model_fn  : str,
                       *args,
                       **kwargs) -> Array:
    """
    Calculates the poissonion likelihood after updating the model using the
    update_fn. The extra *args and **kwargs are passed to the model function.

    Parameters
    ----------
    X : Array
        The array of values to update the model with.
    data : Array
        The data to calulcate the prior with respect to.
    model : object
        The model used to model the data.
    update_fn : Callable
        The function used to update the model before modelling. This should
        take in the X array as the first argument and the model as the second.
    model_fn : str
        The name of the model function to use.

    Returns
    -------
    likelhood : Array
        The poissonian likelihood of the updated model with respect to the data.
    """
    model = update_fn(X, model)
    psf = getattr(model, model_fn)(*args, **kwargs)
    return np.sum(poisson.pmf(data, psf))


def poisson_log_likelihood(X         : Array,
                           data      : Array,
                           model     : object,
                           update_fn : Callable,
                           model_fn  : str,
                           *args,
                           **kwargs) -> Array:
    """
    Calculates the poissonion log likelihood after updating the model using the
    update_fn. The extra *args and **kwargs are passed to the model function.

    Parameters
    ----------
    X : Array
        The array of values to update the model with.
    data : Array
        The data to calulcate the prior with respect to.
    model : object
        The model used to model the data.
    update_fn : Callable
        The function used to update the model before modelling. This should
        take in the X array as the first argument and the model as the second.
    model_fn : str
        The name of the model function to use.

    Returns
    -------
    likelhood : Array
        The poissonian log likelihood of the updated model with respect to the
        data.
    """
    model = update_fn(X, model)
    psf = getattr(model, model_fn)(*args, **kwargs)
    return np.sum(poisson.logpmf(data, psf))


def chi2_likelihood(X         : Array,
                    data      : Array,
                    model     : object,
                    update_fn : Callable,
                    model_fn  : str,
                    *args,
                    **kwargs) -> Array:
    """
    Calculates the chi2 likelihood after updating the model using the
    update_fn. The extra *args and **kwargs are passed to the model function.

    Parameters
    ----------
    X : Array
        The array of values to update the model with.
    data : Array
        The data to calulcate the prior with respect to.
    model : object
        The model used to model the data.
    update_fn : Callable
        The function used to update the model before modelling. This should
        take in the X array as the first argument and the model as the second.
    model_fn : str
        The name of the model function to use.

    Returns
    -------
    likelhood : Array
        The chi2 likelihood of the updated model with respect to the data.
    """
    model = update_fn(X, model)
    psf = getattr(model, model_fn)(*args, **kwargs)
    return np.sum(np.square((psf-data)/noise))


def chi2_log_likelihood(X         : Array,
                        data      : Array,
                        model     : object,
                        update_fn : Callable,
                        model_fn  : str,
                        *args,
                        **kwargs) -> Array:
    """
    Calculates the chi2 log likelihood after updating the model using the
    update_fn. The extra *args and **kwargs are passed to the model function.

    Parameters
    ----------
    X : Array
        The array of values to update the model with.
    data : Array
        The data to calulcate the prior with respect to.
    model : object
        The model used to model the data.
    update_fn : Callable
        The function used to update the model before modelling. This should
        take in the X array as the first argument and the model as the second.
    model_fn : str
        The name of the model function to use.

    Returns
    -------
    likelhood : Array
        The chi2 log likelihood of the updated model with respect to the data.
    """
    model = update_fn(X, model)
    psf = getattr(model, model_fn)(*args, **kwargs)
    return np.log(np.sum(np.square((psf-data)/noise)))


def calculate_covariance(likelihood_fn : Callable,
                         *args,
                         **kwargs) -> Array:
    """
    Calcuates the covariance matrix under the Laplace approximation for the
    given likelihood function. The *args and **kwargs are passed to the
    likelihood function.

    Parameters
    ----------
    likelihood_fn : Callable
        The likelihood function to calculate the covaraince matrix with
        respect to.

    Returns
    -------
    covaraince : Array
        The corresponding covariance matrix.
    """
    matrix = hessian(likelihood_fn)(*args, **kwargs)
    return -np.linalg.inv(matrix)


def calculate_entropy(likelihood_fn : Callable,
                      *args,
                      **kwargs):
    """
    Calcuates the entropy of the covaraince matrix under the Laplace
    approximation for the given likelihood function. The *args and **kwargs
    are passed to through the calculate_covaraince function through to the
    likelihood function.

    Parameters
    ----------
    likelihood_fn : Callable
        The likelihood function to calculate the entropy with respect to.

    Returns
    -------
    entropy : Array
        The entropy of the covariance matrix for the given likelihood function.
    """
    cov = calculate_covariance(likelihood_fn, *args, **kwargs)
    sign, logdet = np.linalg.slogdet(cov)
    return 0.5 * (np.log(2 * np.pi * np.e) + (sign * logdet))