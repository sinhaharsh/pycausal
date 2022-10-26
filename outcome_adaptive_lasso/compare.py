from collections import defaultdict

import pandas as pd
from causallib.estimation import IPW
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy import stats

from simulation import SimulateDataset
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


def calc_ate_ipw(A, Y, X):
    ipw = IPW(LogisticRegression(solver='liblinear', penalty='l1', C=1e2,
                                 max_iter=500), use_stabilized=False).fit(X, A)
    weights = ipw.compute_weights(X, A)
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


def compare_methods():
    simulation = SimulateDataset(num_c=2,
                                 num_p=2,
                                 num_i=2,
                                 num_covariates=7,
                                 coef_c=0.6,
                                 coef_p=0.6,
                                 coef_i=1,
                                 eta=0)
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


def run_multiple_times():
    ate = defaultdict(list)
    for i in range(100):
        estimates = compare_methods()
        for key, value in estimates.items():
            ate[key].append(value)
    return ate


def subplot_violin(data, filename):
    # if not isinstance(data, pd.DataFrame):
    #     data = pd.DataFrame(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.violinplot(x='Method', y='Estimate', data=data, ax=ax, palette=sns.color_palette("Set1"))
    ax.grid()
    ax.set_title('Different estimation alternatives')
    plt.tight_layout()
    fig.savefig("estimation.png", dpi=300)


if __name__ == '__main__':
    results = run_multiple_times()
