from OAL.compare import *
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings, HealthCheck, assume
from hypothesis.extra.numpy import arrays
import pandas as pd


num_samples = st.shared(st.integers(min_value=1, max_value=512))
num_covariates = st.shared(st.integers(min_value=1, max_value=64))
MAX_SIZE = 512


# @st.composite
# def small_shapes(draw, *, ndims=2, max_elems=MAX_SIZE):
#     shape = []
#     for _ in range(ndims):
#         side = draw(st.integers(1, max_elems//5))
#         shape.append(side)
#     if ndims == 1:
#         shape.append(1)
#     return tuple(shape)


@settings(max_examples=5, deadline=None, derandomize=True,
          suppress_health_check=(HealthCheck.too_slow,))
@given(
    A=arrays(dtype=np.float32, shape=st.tuples(num_samples,)),
    Y=arrays(dtype=np.float32, shape=st.tuples(num_samples,)),
    X=arrays(dtype=np.float32, shape=st.tuples(num_samples, num_covariates))
)
def test_calc_vanilla_beta(A, Y, X):
    A = pd.Series(A)
    Y = pd.Series(Y)
    X = pd.DataFrame(X)
    XA = X.merge(A.to_frame(),
                 left_index=True,
                 right_index=True).assign(intecept=1)
    # det = np.linalg.det(XA)
    # assume(not np.isclose(det, 0))

    beta1 = calc_vanilla_beta(A, Y, X)
    beta2 = np.linalg.inv(XA.T.dot(XA)).dot(XA.T.dot(Y)).flatten()[-2]
    assert beta2 == beta1
# print(f"testing shape: {A}")
# assert 1 <= np.prod(A) <= MAX_SIZE
