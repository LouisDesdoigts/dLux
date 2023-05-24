<!-- # Bayesian Utility Functions

This module contains a few likelihood functions. plus functions to calcualte things like covariance matrices and its entropy. This module is likely to be moved in the base package of dLux, [Zodiax](https://github.com/LouisDesdoigts/zodiax).

These functions have an unusual syntax becuase they are designed for calculating [Fisher Matrices](https://en.wikipedia.org/wiki/Fisher_information). To get an idea of how to use them please refer to the [Fisher Matrix Tutorial](../tutorials/fisher_matrix.ipynb). Later these docs will be expanded with specific examples.

## Poisson Likelihood

Calculates the Poisson likelihood after updating some model with the `update_fn` and calling the `model_fn`.

??? info "Poisson Likelihood API"
    ::: dLux.utils.bayes.poisson_likelihood

## Poisson Log Likelihood

Calculates the Poisson log likelihood after updating some model with the `update_fn` and calling the `model_fn`.

??? info "Poisson Log Likelihood API"
    ::: dLux.utils.bayes.poisson_log_likelihood

## Chi2 Likelihood

Calculates the Chi2 likelihood after updating some model with the `update_fn` and calling the `model_fn`.

??? info "Chi2 Likelihood API"
    ::: dLux.utils.bayes.chi2_likelihood

## Chi2 Log Likelihood

Calculates the Chi2 log likelihood after updating some model with the `update_fn` and calling the `model_fn`.

??? info "Chi2 Log Likelihood API"
    ::: dLux.utils.bayes.chi2_log_likelihood

## Calculate Covariance

Calcluates the covaraince matrix of some likelihood function.

??? info "Calculate Covariance API"
    ::: dLux.utils.bayes.calculate_covariance

## Calculate Entropy

Calculates the entropy of a covariance matrix.

??? info "Calculate Entropy API"
    ::: dLux.utils.bayes.calculate_entropy -->