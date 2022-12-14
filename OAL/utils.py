import warnings

import numpy as np
from pathlib import Path
import json
import time


def timestamp() -> str:
    """
    Generates time string in the specified format
    @return: time cast as string
    """
    time_string = time.strftime("%m_%d_%Y_%H_%M")
    return time_string


def save_dict2json(folder, filename, dictionary):
    if not Path(folder).exists():
        Path(folder).mkdir(parents=True)
    fullpath = Path(folder)/(filename+'.json')
    with open(fullpath, 'w') as fp:
        json.dump(dictionary, fp, indent=4)


def print_corr(df):
    A = df.pop('A')
    Y = df.pop('Y')
    X = df

    print('Correlation with treatment : ')
    for col in X.columns:
        print("A - ", col, ' : ', A.corr(X[col]))

    print('Correlation with outcome : ')
    for col in X.columns:
        print("Y - ", col, ' : ', Y.corr(X[col]))


def get_psd_matrix(size, diagonal=1):
    # computing the nearest correlation matrix (i.e., enforcing unit diagonal).
    # Higham 2000
    # https://www.maths.manchester.ac.uk/~higham/narep/narep369.pdf
    # https://stackoverflow.com/a/10940283/3140172
    while True:
        A = 0.01*np.random.rand(size, size)
        B = np.dot(A, A.transpose())
        np.fill_diagonal(B, diagonal)
        C = near_pd(B, nit=100)
        if np.all(np.linalg.eigvals(C) > 0):
            return C
        else:
            # It is possible that eigvalues are -0.000 due to floating
            # point error. Need to correct for it.
            min_eigvalue = min(np.linalg.eigvals(C))
            C -= 10*min_eigvalue*np.eye(*C.shape)
            if np.all(np.linalg.eigvals(C) > 0):
                return C
        warnings.warn("regenerating random correlation matrix")


def _get_a_plus(matrix):
    eigval, eigvec = np.linalg.eig(matrix)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T


def _get_ps(matrix, W=None):
    W05 = np.matrix(W**.5)
    return W05.I * _get_a_plus(W05 * matrix * W05) * W05.I


def _get_pu(matrix, W=None):
    Aret = np.array(matrix.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)


def near_pd(matrix, nit=10):
    n = matrix.shape[0]
    W = np.identity(n)
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = matrix.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _get_ps(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _get_pu(Xk, W=W)
    return Yk


def check_overlap(data1, data2):
    bin_edges = np.histogram_bin_edges(np.concatenate([data1, data2]))
    hist1, _ = np.histogram(data1, bins=bin_edges)
    hist2, _ = np.histogram(data2, bins=bin_edges)
    return np.minimum(hist1, hist2).sum() > 10


if __name__ == '__main__':
    psd = get_psd_matrix(4, 1)
