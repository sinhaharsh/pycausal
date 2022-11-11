from compare import run_multiple_times
import warnings
warnings.filterwarnings("ignore")

situation1 = {
    'num_samples': 200,
    'num_covariates': 100
}
situation2 = {
    'num_samples': 500,
    'num_covariates': 200
}
situation3 = {
    'num_samples': 20,
    'num_covariates': 200
}
situation4 = {
    'num_samples': 20,
    'num_covariates': 500
}
situation5 = {
    'num_samples': 20,
    'num_covariates': 1000
}
situation_values = [situation1,
                    situation2,
                    situation3,
                    situation4,
                    situation5]
eta_values = [0, 2]
rho_values = [0, 0.2, 0.5]
for eta in eta_values:
    for rho in rho_values:
        for situation in situation_values:
            params = {
                'num_c': 2,
                'num_p': 2,
                'num_i': 2,
                'num_covariates': situation['num_covariates'],
                'num_samples': situation['num_samples'],
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
                'num_covariates': situation['num_covariates'],
                'num_samples': situation['num_samples'],
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
                'num_covariates': situation['num_covariates'],
                'num_samples': situation['num_samples'],
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
                'num_covariates': situation['num_covariates'],
                'num_samples': situation['num_samples'],
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
