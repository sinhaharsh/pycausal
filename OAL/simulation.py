import warnings
from typing import Union, List

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from OAL.utils import get_psd_matrix


class SimulateDataset:
    def __init__(self,
                 num_c,
                 num_p,
                 num_i,
                 num_covariates,
                 num_samples,
                 coef_c: Union[float, List[float]],
                 coef_p,
                 coef_i,
                 eta,
                 rho):
        self.num_samples = num_samples
        self.num_c = num_c
        self.num_p = num_p
        self.num_i = num_i
        self.rho = rho
        self.num_covariates = num_covariates
        self.num_s = self.num_covariates - num_c - num_p - num_i
        self.data = None
        self.train_data = None
        self.test_data = None
        if isinstance(coef_c, list):
            self.coef_c = coef_c
        else:
            self.coef_c = [coef_c, coef_c]
        self.coef_p = coef_p
        self.coef_i = coef_i
        self.eta = eta
        if self.num_s < 0:
            raise SyntaxError('the number of spurious variables < 0')

    def generate_col_names(self):
        """Generate column names for the synthetic dataset """

        def make_names(count, subscript):
            return ['X{1}{0}'.format(i, subscript) for i in range(1, count + 1)]

        pC = self.num_c  # number of confounders
        pP = self.num_p  # number of outcome predictors/ Precision variables
        pI = self.num_i  # number of exposure predictors/ Instrumental variables
        pS = self.num_s  # number of spurious variables

        col_names = ['A', 'Y'] + make_names(pC, 'c') + make_names(pP, 'p') \
                    + make_names(pI, 'i') + make_names(pS, 's')
        return col_names

    def load_scenario(self):
        """Utility function to load predefined scenarios"""
        c_indexes = list(range(self.num_c))
        p_indexes = list(range(self.num_c,
                               self.num_c + self.num_p))
        i_indexes = list(range(self.num_p + self.num_c,
                               self.num_p + self.num_c + self.num_i))
        nu = np.zeros(self.num_covariates)
        beta = np.zeros(self.num_covariates)
        beta[c_indexes] = self.coef_c[0]
        beta[p_indexes] = self.coef_p
        nu[c_indexes] = self.coef_c[1]
        nu[i_indexes] = self.coef_i
        return beta, nu

    def generate_dataset(self):
        susan_rho = True
        while True:
            # covariance matrix of the Gaussian covariates.
            if not susan_rho:
                cov_x = get_psd_matrix(self.num_covariates, diagonal=1)
            else:
                cov_x = np.eye(self.num_covariates) + \
                        ~np.eye(self.num_covariates, dtype=bool) * self.rho

            X = np.random.multivariate_normal(
                mean=0 * np.ones(self.num_covariates),
                cov=cov_x,
                size=self.num_samples)
            # Normalize covariates to have 0 mean unit std
            scaler = StandardScaler(copy=False)
            scaler.fit_transform(X)

            # Load beta and nu from the predefined scenarios
            beta, nu = self.load_scenario()
            A = np.random.binomial(np.ones(self.num_samples, dtype=int),
                                   expit(np.dot(X, nu)))

            if np.all(A == A[0]):
                warnings.warn(
                    "Regenerating treatment to ensure atleast one sample for both classes - 0 and 1.")
            else:
                break

        Y = np.random.randn(self.num_samples) + self.eta * A + np.dot(X, beta)
        col_names = self.generate_col_names()
        df = pd.DataFrame(np.hstack([A.reshape(-1, 1), Y.reshape(-1, 1), X]),
                          columns=col_names)
        self.data = df

    def create_train_test(self, test_size=0.3):
        # train, test = train_test_split(self.data,
                                       # test_size=test_size)
        self.train_data = self.get_partition(self.data.copy())
        self.test_data = self.get_partition(self.data.copy())

    @staticmethod
    def get_partition(data):
        splits = {}
        splits['A'] = data.pop('A')
        splits['Y'] = data.pop('Y')
        splits['X_conf'] = data[[col for col in data if col.startswith('Xc')]]
        splits['X_target'] = data[
            [col for col in data if col.startswith('Xc')] +
            [col for col in data if col.startswith('Xp')]]
        splits['X_pot_conf'] = data[
            [col for col in data if col.startswith('Xc')] +
            [col for col in data if col.startswith('Xp')] +
            [col for col in data if col.startswith('Xi')]]
        splits['X'] = data[[col for col in data if col.startswith('X')]]
        return splits
