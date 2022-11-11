from compare import run_multiple_times
import warnings
warnings.filterwarnings("ignore")

num_samples = [200, 500, 1000, 2000]
num_covariates = [100, 200, 500, 1000]
eta_values = [-2, 0, 2]
rho_values = [0, 0.2, 0.5]
for eta in eta_values:
    for rho in rho_values:
        for ns in num_samples :
            for nc in num_covariates:
                params = {
                    'num_c': 2,
                    'num_p': 2,
                    'num_i': 2,
                    'num_covariates': nc,
                    'num_samples': ns,
                    'coef_c': [0.6, 1],
                    'coef_p': 0.6,
                    'coef_i': 1,
                    'eta': eta,
                    'rho': rho,
                    'solver': 'liblinear',
                    'C': 1e-1,
                    'max_iter': 2000,
                    'penalty': 'l1',
                    'scenario': 1,
                    'plot': 'box'
                }
                results_scenario1 = run_multiple_times(params, visualize=True)

                params = {
                    'num_c': 2,
                    'num_p': 2,
                    'num_i': 2,
                    'num_covariates': nc,
                    'num_samples': ns,
                    'coef_c': [0.6, 0.4],
                    'coef_p': 0.6,
                    'coef_i': 1,
                    'eta': eta,
                    'rho': rho,
                    'solver': 'liblinear',
                    'C': 1e-1,
                    'max_iter': 500,
                    'penalty': 'l1',
                    'scenario': 2,
                    'plot': 'box'
                }
                results_scenario2 = run_multiple_times(params, visualize=True)

                params = {
                    'num_c': 2,
                    'num_p': 2,
                    'num_i': 2,
                    'num_covariates': nc,
                    'num_samples': ns,
                    'coef_c': [0.2, 0.4],
                    'coef_p': 0.6,
                    'coef_i': 1,
                    'eta': eta,
                    'rho': rho,
                    'solver': 'liblinear',
                    'C': 1e-1,
                    'max_iter': 500,
                    'penalty': 'l1',
                    'scenario': 3,
                    'plot': 'box'
                }
                results_scenario3 = run_multiple_times(params, visualize=True)

                params={
                    'num_c': 2,
                    'num_p': 2,
                    'num_i': 2,
                    'num_covariates': nc,
                    'num_samples': ns,
                    'coef_c': [0.6, 1],
                    'coef_p': 0.6,
                    'coef_i': 1.8,
                    'eta': eta,
                    'rho': rho,
                    'solver': 'liblinear',
                    'C': 1e-1,
                    'max_iter': 500,
                    'penalty': 'l1',
                    'scenario': 4,
                    'plot': 'box'
                }
                results_scenario4 = run_multiple_times(params, visualize=True)
