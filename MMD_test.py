# Computes RBF-kernel MMD^2 between two groups (e.g., GOOD vs BAD) with median-heuristic gamma
# and an optional permutation test (p-value).

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

def median_heuristic_gamma(X, Y=None, sample_max=2000):
    """Return RBF gamma = 1 / median(pairwise_sq_dists) using a subsample if large."""
    if Y is None:
        Z = np.asarray(X, dtype=float)
    else:
        Z = np.vstack([X, Y]).astype(float)
    n = Z.shape[0]
    if n > sample_max:
        idx = np.random.RandomState(0).choice(n, sample_max, replace=False)
        Zs = Z[idx]
    else:
        Zs = Z
    dists = pdist(Zs, metric='sqeuclidean')
    med = np.median(dists) if len(dists) > 0 else 1.0
    return 1.0 / (med + 1e-12)

def mmd_rbf_unbiased(X, Y, gamma=None):
    """Unbiased estimator of MMD^2 with RBF kernel.
    Returns (mmd2, gamma_used).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n = X.shape[0]; m = Y.shape[0]
    if n < 2 or m < 2:
        return np.nan, None
    if gamma is None:
        gamma = median_heuristic_gamma(X, Y)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    sum_kxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sum_kyy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sum_kxy = Kxy.sum() / (n * m)
    mmd2 = sum_kxx + sum_kyy - 2.0 * sum_kxy
    return float(mmd2), float(gamma)

def mmd_permutation_test(X, Y, n_permutations=200, gamma=None, seed=0):
    """Permutation test for MMD. Returns (mmd2_obs, gamma, p_value)."""
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if Y.ndim == 1:
        Y = Y.reshape(-1,1)
    n = X.shape[0]; m = Y.shape[0]
    if n < 2 or m < 2:
        return np.nan, None, np.nan
    Z = np.vstack([X, Y])
    labels = np.array([0]*n + [1]*m)
    if gamma is None:
        gamma = median_heuristic_gamma(X, Y)
    mmd_obs, _ = mmd_rbf_unbiased(X, Y, gamma=gamma)
    perm_stats = []
    for i in range(n_permutations):
        rng.shuffle(labels)
        Xp = Z[labels==0]
        Yp = Z[labels==1]
        # If permutation produced unequal sizes (shouldn't), sample to original sizes
        if Xp.shape[0] != n:
            idx0 = np.where(labels==0)[0][:n]
            idx1 = np.where(labels==1)[0][:m]
            Xp = Z[idx0]
            Yp = Z[idx1]
        stat, _ = mmd_rbf_unbiased(Xp, Yp, gamma=gamma)
        perm_stats.append(stat)
    perm_stats = np.array(perm_stats)
    p_value = (np.sum(perm_stats >= mmd_obs) + 1) / (len(perm_stats) + 1)
    return float(mmd_obs), float(gamma), float(p_value)



    