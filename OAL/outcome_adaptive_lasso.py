from math import log

import numpy as np
import pandas as pd
from causallib.estimation import IPW
from sklearn.linear_model import LogisticRegression, LinearRegression


def check_input(A, Y, X):
    if not isinstance(A, pd.Series):
        if not np.max(A.shape) == A.size:
            raise Exception(f'A must be one dimensional, got shape {A.shape}')
        A = pd.Series(A.flatten())
    if not isinstance(Y, pd.Series):
        if not np.max(A.shape) == A.size:
            raise Exception(f'A must be one dimensional, got shape {A.shape}')
        Y = pd.Series(Y.flatten())
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not len(A.index) == len(Y.index) == len(X.index):
        raise Exception(f'A, Y, X must have same number of samples, '
                        f'got A: {len(A.index)} samples, '
                        f'Y: {len(Y.index)} samples, X: {len(X.index)} samples')
    return A, Y, X


def calc_ate_vanilla_ipw(A, Y, X):
    ipw = IPW(LogisticRegression(solver='liblinear', penalty='l1', C=1e2,
                                 max_iter=500), use_stabilized=True).fit(X, A)
    weights = ipw.compute_weights(X, A)
    outcomes = ipw.estimate_population_outcome(X, A, Y, w=weights)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0])
    return effect[0]


def calc_group_diff(X, idx_trt, ipw, l_norm):
    """
    Utility function to calculate the difference in covariates between
    treatment and control groups
    """
    return (np.abs(np.average(X[idx_trt], weights=ipw[idx_trt], axis=0) -
                   np.average(X[~idx_trt], weights=ipw[~idx_trt],
                              axis=0))) ** l_norm


def calc_wamd(A, X, ipw, x_coefs, l_norm=1):
    """Utility function to calculate the weighted absolute mean difference"""
    idx_trt = A == 1
    return calc_group_diff(X.values, idx_trt.values, ipw.values, l_norm).dot(
        np.abs(x_coefs))


def calc_oal_single_lambda(data, log_lambda, gamma_factor):
    """Calculate ATE with the outcome adaptive lasso"""
    train = data.train_data
    test = data.test_data
    A_train = train['A']
    Y_train = train['Y']
    X_train = train['X']

    A_test = test['A']
    Y_test = test['Y']
    X_test = test['X']

    A_train, Y_train, X_train = check_input(A_train, Y_train, X_train)

    n = A_train.shape[0]  # number of samples
    _lambda = n ** log_lambda
    # extract gamma according to _lambda and gamma_factor
    gamma = 2 * (1 + gamma_factor - log(_lambda, n))

    # fit regression from covariates X and exposure A to outcome Y
    XA_train = X_train.merge(A_train.to_frame(),
                             left_index=True,
                             right_index=True)
    lr = LinearRegression(fit_intercept=True).fit(XA_train, Y_train)


    # extract the coefficients of the covariates
    x_coefs = lr.coef_.flatten()[:-1]

    # calculate outcome adaptive penalization weights
    weights = (np.abs(x_coefs)) ** (-gamma)

    # apply the penalization to the covariates themselves
    X_w = X_train / weights

    # fit logistic propensity score model from penalized covariates
    # to the exposure
    ipw = IPW(
        LogisticRegression(solver='liblinear',
                           penalty='l1',
                           C=1 / _lambda,
                           max_iter=400),
        use_stabilized=False).fit(X_w, A_train)
    # compute inverse propensity weighting and calculate ATE
    weights_train = ipw.compute_weights(X_train, A_train)
    weights_test = ipw.compute_weights(X_test, A_test)
    outcomes = ipw.estimate_population_outcome(X_test, A_test, Y_test,
                                               w=weights_test)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0])
    wamd = calc_wamd(A_train, X_train, weights_train, x_coefs)
    return effect, wamd


def calc_outcome_adaptive_lasso(data, gamma_factor=2,
                                log_lambdas=None):
    """
    Calculate estimate of average treatment effect using the outcome adaptive
     LASSO (Shortreed and Ertefaie, 2017)

    Parameters
    ----------
    A : Exposure (=treatment, intervention)
     pandas series or one-dimensional numpy array
    Y : Outcome
     pandas series or one-dimensional numpy array
    X : Covariates
     pandas dataframe or two-dimensional numpy array
     (shape n_samples, n_covariates)
    log_lambdas : log of lambda - strength of adaptive LASSO regularization.
     If log_lambdas has multiple values, lambda will be selected according to
     the minimal absolute mean difference, as suggested in the paper
     If None, it will be set to the suggested search list in the paper:
     [-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49]
    gamma_factor : a constant to couple between lambda and gamma,
     the single-feature penalization strength. The equation relating gamma and
     lambda is lambda * n^(gamma/2 -1) = n^gamma_factor
     Default value is 2, as suggested in the paper for the synthetic
     dataset experiments

    Returns
    -------
    ate : estimate of the average treatment effect
    """


    if log_lambdas is None:
        log_lambdas = [-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49]
    # n = A.shape[0]
    lambdas = np.array(log_lambdas)
    amd_vec = np.zeros(lambdas.shape[0])
    ate_vec = np.zeros(lambdas.shape[0])

    # Calculate ATE for each lambda,
    # select the one minimizing the weighted absolute mean difference
    for il in range(len(lambdas)):
        ate_vec[il], amd_vec[il] = calc_oal_single_lambda(data,
                                                           lambdas[il],
                                                           gamma_factor)
        # print("Coeff of first 8 covariates : {}".format(x_coefs[:8]))
    return ate_vec[np.nanargmin(amd_vec)]


if __name__ == '__main__':
    from synthetic_data_simulation import generate_synthetic_dataset

    for eta in range(0, 20, 2):
        print(f"True Estimate : {eta}")
        df = generate_synthetic_dataset(n=200, d=100, rho=0, eta=eta, scenario_num=4)
        A = df.pop('A')
        Y = df.pop('Y')
        X = df
        ate = calc_outcome_adaptive_lasso(A, Y, X)
        ate_vanilla = calc_ate_vanilla_ipw(A, Y, X)
        print("OAL Estimate : {}\t IPW Estimate : {}".format(ate, ate_vanilla))