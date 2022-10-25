import numpy as np


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


def get_psd_matrix(size, diagonal=None):
    A = np.random.rand(size, size)
    B = np.dot(A, A.transpose())
    if diagonal:
        np.fill_diagonal(B, diagonal)
    assert np.allclose(B, B.transpose())
    return B
