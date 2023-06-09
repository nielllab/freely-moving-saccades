import numpy as np

def z_score(A):
    return (np.max(np.abs(A))-np.mean(A)) / np.std(A)


def stderr(A):
    return np.std(A) / np.sqrt(len(A))


def drop_nan_along(x, axis=1):
    # axis=1 will drop along columns (i.e. any rows with NaNs will be dropped)
    x = x[~np.isnan(x).any(axis=axis)]
    return x