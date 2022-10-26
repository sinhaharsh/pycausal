from collections import defaultdict

import pandas as pd
from causallib.estimation import IPW
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy import stats

from simulation import SimulateDataset
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from utils import timestamp, save_dict2json
from pathlib import Path


OUT_DIR = './data/'

def calc_ate_ipw(A, Y, X):
    ipw = IPW(LogisticRegression(solver='liblinear', penalty='l1', C=1e2,
                                 max_iter=500), use_stabilized=False).fit(X, A)
    weights = ipw.compute_weights(X, A, treatment_values=1)
    outcomes = ipw.estimate_population_outcome(X, A, Y, w=weights)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0])
    return effect[0]


def calc_vanilla_beta(A, Y, X):
    # fit regression from covariates X and exposure A to outcome Y
    XA = X.merge(A.to_frame(), left_index=True, right_index=True)
    lr = LinearRegression(fit_intercept=True).fit(XA, Y)

    # extract the coefficients of the covariates
    coef = lr.coef_.flatten()[-1]
    return coef


def compare_methods(num_c=2,
                    num_p=2,
                    num_i=2,
                    num_covariates=7,
                    coef_c=0.6,
                    coef_p=0.6,
                    coef_i=1,
                    eta=0):
    simulation = SimulateDataset(num_c=num_c,
                                 num_p=num_p,
                                 num_i=num_i,
                                 num_covariates=num_covariates,
                                 coef_c=coef_c,
                                 coef_p=coef_p,
                                 coef_i=coef_i,
                                 eta=eta)
    dataset = simulation.generate_dataset()
    A = dataset.pop('A')
    Y = dataset.pop('Y')
    X_conf = dataset[[col for col in dataset if col.startswith('Xc')]]
    X_target = dataset[[col for col in dataset if col.startswith('Xc')] +
                       [col for col in dataset if col.startswith('Xp')]]
    X_pot_conf = dataset[[col for col in dataset if col.startswith('Xc')] +
                         [col for col in dataset if col.startswith('Xp')] +
                         [col for col in dataset if col.startswith('Xi')]]
    X_all = dataset[[col for col in dataset if col.startswith('X')]]
    results = {
        'regression': calc_vanilla_beta(A, Y, X_all),
        'conf': calc_ate_ipw(A, Y, X_conf),
        'target': calc_ate_ipw(A, Y, X_target),
        'pot_conf': calc_ate_ipw(A, Y, X_pot_conf),
        'all': calc_ate_ipw(A, Y, X_all)
    }
    return results


def run_multiple_times(visualize=False):
    ate = list()
    filename = timestamp()
    params = {
        'num_c': 2,
        'num_p': 2,
        'num_i': 2,
        'num_covariates': 200,
        'coef_c': [0.7, 0.5],
        'coef_p': 0.8,
        'coef_i': 0.9,
        'eta': 0
    }

    for i in range(100):
        estimates = compare_methods(
            params['num_c'], params['num_p'], params['num_i'],
            params['num_covariates'], params['coef_c'],
            params['coef_p'], params['coef_i'], params['eta']
        )

        ate.extend(estimates.items())
    ate_df = pd.DataFrame(ate, columns=['Method', 'Estimate'])
    if visualize:
        subplot_violin(ate_df, OUT_DIR, filename)
    save_dict2json(OUT_DIR, filename, params)
    return ate


def subplot_violin(data, folder, filename):
    # if not isinstance(data, pd.DataFrame):
    #     data = pd.DataFrame(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.violinplot(x='Method', y='Estimate', data=data, ax=ax, palette=sns.color_palette("Set1"))
    ax.grid()
    ax.set_title('Different estimation alternatives')
    plt.tight_layout()

    if not Path(folder).exists():
        Path(folder).mkdir(parents=True)
    fullpath = Path(folder) / (filename + '.png')

    fig.savefig(fullpath, dpi=300)


if __name__ == '__main__':
    results = run_multiple_times(visualize=True)
