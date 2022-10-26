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
from utils import check_overlap


OUT_DIR = './data/'


def calc_ate_ipw(A, Y, X, solver='liblinear', C=1e-2, max_iter=500):
    ipw = IPW(LogisticRegression(solver=solver, penalty='l1', C=C,
                                 max_iter=max_iter), use_stabilized=True).fit(X, A)
    if check_balance(A, Y, X, ipw, visualize=False):
        print(f"Num_features : {X.shape[0]}, No overlap, IPW cannot be estimated.")
        return np.NAN
    weights = ipw.compute_weights(X, A, treatment_values=1, clip_min=0.2, clip_max=0.8)
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


def check_balance(A, Y, X, ipw, folder=OUT_DIR, visualize=False):
    propensity = ipw.compute_propensity(X, A, treatment_values=1)
    AP = pd.concat({'A': A,
                    'P': propensity}, axis=1)
    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.displot(AP, x="P", hue="A", kind="hist", multiple="stack")
        ax.grid()
        ax.set_title('Distributional balance')
        plt.tight_layout()
        if not Path(folder).exists():
            Path(folder).mkdir(parents=True)
        fullpath = Path(folder) / (timestamp() + '.png')

        plt.savefig(fullpath, dpi=300)
        plt.close(fig)
    if check_overlap(AP.loc[AP['A'] == 1, 'P'],
                     AP.loc[AP['A'] == 0, 'P']):
        return True
    return False



def compare_methods(num_c, num_p, num_i,
                    num_covariates,
                    coef_c, coef_p, coef_i,
                    eta, solver='liblinear', C=1e-2, max_iter=500):
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
        'conf': calc_ate_ipw(A, Y, X_conf, solver=solver, C=C, max_iter=max_iter),
        'target': calc_ate_ipw(A, Y, X_target, solver=solver, C=C, max_iter=max_iter),
        'pot_conf': calc_ate_ipw(A, Y, X_pot_conf, solver=solver, C=C, max_iter=max_iter),
        'all': calc_ate_ipw(A, Y, X_all, solver=solver, C=C, max_iter=max_iter)
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
        'eta': 0,
        'solver': 'liblinear',
        'C': 1e-1,
        'max_iter': 500
    }

    for i in range(100):
        estimates = compare_methods(**params)

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
