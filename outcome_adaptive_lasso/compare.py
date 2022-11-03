from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from causallib.estimation import IPW
from sklearn.linear_model import LogisticRegression, LinearRegression

from simulation import SimulateDataset
from utils import check_overlap
from utils import timestamp, save_dict2json

OUT_DIR = './susan/'


def calc_ate_ipw(A, Y, X,
                 solver='liblinear', penalty='l1', C=1e-2, max_iter=500):
    ipw = IPW(LogisticRegression(solver=solver, penalty=penalty, C=C,
                                 max_iter=max_iter), use_stabilized=True).fit(X,
                                                                              A)
    # if not check_balance(A, Y, X, ipw, visualize=False):
    #     print(
    #         f"Num_features : {X.shape[1]}, No overlap, IPW cannot be estimated.")
    #     return np.NAN
    weights = ipw.compute_weights(X, A,
                                  treatment_values=1)  # , clip_min=0.2, clip_max=0.8)
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
                    num_covariates, num_samples,
                    coef_c, coef_p, coef_i,
                    eta, rho, solver='liblinear', penalty='l1',
                    C=1e-2, max_iter=500, **kwargs):
    simulation = SimulateDataset(num_c=num_c,
                                 num_p=num_p,
                                 num_i=num_i,
                                 num_covariates=num_covariates,
                                 num_samples=num_samples,
                                 coef_c=coef_c,
                                 coef_p=coef_p,
                                 coef_i=coef_i,
                                 eta=eta,
                                 rho=rho)
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
        'conf': calc_ate_ipw(A, Y, X_conf, penalty=penalty,
                             solver=solver, C=C, max_iter=max_iter),
        'target': calc_ate_ipw(A, Y, X_target, penalty=penalty,
                               solver=solver, C=C, max_iter=max_iter),
        'pot_conf': calc_ate_ipw(A, Y, X_pot_conf, penalty=penalty,
                                 solver=solver, C=C, max_iter=max_iter),
        'all': calc_ate_ipw(A, Y, X_all, penalty=penalty,
                            solver=solver, C=C, max_iter=max_iter)
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
        'C': 1e2,
        'max_iter': 500,
        'penalty': 'l1',
    }

    for i in range(100):
        estimates = compare_methods(**params)

        ate.extend(estimates.items())
    ate_df = pd.DataFrame(ate, columns=['Method', 'Estimate'])
    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        subplot_violin(ate_df, OUT_DIR, filename, fig, ax)
    save_dict2json(OUT_DIR, filename, params)
    plt.tight_layout()

    if not Path(OUT_DIR).exists():
        Path(OUT_DIR).mkdir(parents=True)
    fullpath = Path(OUT_DIR) / (filename + '.png')

    fig.savefig(fullpath, dpi=300)
    plt.close(fig)
    return ate

def vary_eta(visualize=False):

    possible_values = range(-6, 6, 2)
    nrow = 2
    ncol = 3
    fig, ax = plt.subplots(nrow, ncol, figsize=(16, 16))
    filename = timestamp()
    for j, eta in enumerate(possible_values):
        ate = list()
        params = {
            'num_c': 5,
            'num_p': 2,
            'num_i': 2,
            'num_covariates': 100,
            'coef_c': [0.01, 0.02],
            'coef_p': 0.3,
            'coef_i': 0.1,
            'eta': eta,
            'solver': 'liblinear',
            'C': 1e-1,
            'max_iter': 500,
            'penalty': 'l1',
        }
        for i in range(100):
            estimates = compare_methods(**params)
            ate.extend(estimates.items())
        ate_df = pd.DataFrame(ate, columns=['Method', 'Estimate'])
        if visualize:
            true_ate = eta
            multiplot_violin(ate_df, true_ate, filename, fig, ax[j//ncol, j % ncol])

    save_dict2json(OUT_DIR, filename, params)
    plt.tight_layout()

    if not Path(OUT_DIR).exists():
        Path(OUT_DIR).mkdir(parents=True)
    fullpath = Path(OUT_DIR) / (filename + '.png')

    fig.savefig(fullpath, dpi=300)
    plt.close(fig)
    return ate


def multiplot_violin(data, true_ate, filename, fig, ax):
    # if not isinstance(data, pd.DataFrame):
    #     data = pd.DataFrame(data)
    sns.violinplot(x='Method', y='Estimate', data=data,
                   ax=ax, palette=sns.color_palette("Set1"),
                   inner='quartile')
    sns.swarmplot(x='Method', y='Estimate', data=data,
                  ax=ax, color="white")

    ax.grid()
    # ax.set_title('Different estimation alternatives')
    lines = ax.get_lines()
    categories = range(len(lines)//3)

    for cat in categories:
        value_list = lines[1 + cat * 3].get_ydata()
        if len(value_list) > 1:
            y = round(value_list[1], 2)
            ax.text(
                cat,
                y,
                f'{y}',
                ha='center',
                va='center',
                fontweight='bold',
                size=10,
                color='white',
                bbox=dict(facecolor='#445A64'))

    ax.axhline(y=true_ate, color='r', linestyle='-')


def subplot_violin(data, folder, filename, fig, ax):
    # if not isinstance(data, pd.DataFrame):
    #     data = pd.DataFrame(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.violinplot(x='Method', y='Estimate', data=data,
                   ax=ax, palette=sns.color_palette("Set1"),
                   inner='quartile')
    ax.grid()
    ax.set_title('Different estimation alternatives')
    lines = ax.get_lines()
    categories = range(len(lines)//3)

    for cat in categories:
        value_list = lines[1 + cat * 3].get_ydata()
        if len(value_list) > 1:
            y = round(value_list[1], 2)
            ax.text(
                cat,
                y,
                f'{y}',
                ha='center',
                va='center',
                fontweight='bold',
                size=10,
                color='white',
                bbox=dict(facecolor='#445A64'))



if __name__ == '__main__':
    results = run_multiple_times(visualize=True)
