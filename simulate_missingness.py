# simulate_missingness.py
"""
Helpers to simulate MAR and MNAR selection (acceptance) mechanisms, run an
acceptance pass, and run diagnostics to check whether selection is MAR or MNAR.

Usage:
- Provide a pandas DataFrame `df` with feature columns (X_cols) and a binary label
  column 'BAD' (1=bad, 0=good).
- Call selection functions to get a boolean mask `accepted`.
- Use diagnostics to confirm whether selection depends on Y after conditioning on X.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def topk_selection(df, score_col, top_percent):
    """Deterministic selection: accept top_percent by score_col."""
    n = len(df)
    k = max(1, int(np.floor(top_percent * n)))
    idx = np.argsort(df[score_col].values)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

def probabilistic_selection_from_score(df, score, rng, stochastic=True):
    """Convert score to probability via sigmoid and sample (or threshold deterministically)."""
    logits = score - np.mean(score)
    probs = 1.0 / (1.0 + np.exp(-logits))
    if stochastic:
        return rng.random(len(probs)) < probs, probs
    else:
        return probs >= np.median(probs), probs

def selection_mar_logistic(df, X_cols, rng, beta_x=None, intercept=0.0,
                           top_percent=None, stochastic=True):
    """
    MAR selection: P(R=1 | X) via logistic on X.
    - beta_x: array-like; if None random weights are used.
    - top_percent: if provided, do deterministic top-percent selection by score instead.
    Returns (mask_accepted (bool array), probs or None if top_percent used)
    """
    X = df[X_cols].values.astype(float)
    n, k = X.shape
    if beta_x is None:
        beta_x = rng.normal(scale=1.0, size=k)
    score = X @ np.asarray(beta_x) + intercept
    if top_percent is not None:
        df2 = df.copy()
        df2['_score'] = score
        return topk_selection(df2, '_score', top_percent), None
    mask, probs = probabilistic_selection_from_score(df, score, rng, stochastic=stochastic)
    return mask, probs

def selection_mnar_on_y(df, X_cols, rng, beta_x=None, beta_y=1.0, intercept=0.0,
                       top_percent=None, stochastic=True):
    """
    MNAR selection that depends explicitly on the true label Y ('BAD').
    P(R=1 | X, Y) = sigmoid(X @ beta_x + beta_y * Y + intercept)
    """
    X = df[X_cols].values.astype(float)
    n, k = X.shape
    if beta_x is None:
        beta_x = rng.normal(scale=1.0, size=k)
    y = df['BAD'].values.astype(float)
    score = X @ np.asarray(beta_x) + beta_y * y + intercept
    if top_percent is not None:
        df2 = df.copy()
        df2['_score'] = score
        return topk_selection(df2, '_score', top_percent), None
    mask, probs = probabilistic_selection_from_score(df, score, rng, stochastic=stochastic)
    return mask, probs

def selection_mnar_on_latent(df, X_cols, rng, beta_x=None, gamma=1.0, intercept=0.0,
                             top_percent=None, stochastic=True, latent_noise_scale=1.0):
    """
    MNAR through an unobserved latent U correlated with Y:
    U = w^T X + eps  (eps unobserved)
    Selection depends on U (unobserved), making it MNAR in practice.
    """
    X = df[X_cols].values.astype(float)
    n, k = X.shape
    if beta_x is None:
        beta_x = rng.normal(scale=1.0, size=k)
    latent = X @ np.asarray(beta_x) + rng.normal(scale=latent_noise_scale, size=n)
    score = gamma * latent + intercept
    if top_percent is not None:
        df2 = df.copy()
        df2['_score'] = score
        return topk_selection(df2, '_score', top_percent), None
    mask, probs = probabilistic_selection_from_score(df, score, rng, stochastic=stochastic)
    return mask, probs

def selection_mixed(df, X_cols, rng, alpha=0.5, beta_x=None, beta_y=1.0, intercept=0.0, stochastic=True):
    """
    Mixed MAR + MNAR: convex combination of a selection score that uses X and one that uses Y.
    alpha in [0,1] how much weight to the Y-dependent part (MNAR). alpha=0 => MAR, alpha=1 => full MNAR.
    """
    X = df[X_cols].values.astype(float)
    n, k = X.shape
    if beta_x is None:
        beta_x = rng.normal(scale=1.0, size=k)
    score_x = X @ np.asarray(beta_x)
    score_y = beta_y * df['BAD'].values.astype(float)
    score = (1.0 - alpha) * score_x + alpha * score_y + intercept
    mask, probs = probabilistic_selection_from_score(df, score, rng, stochastic=stochastic)
    return mask, probs

# Diagnostics
def test_mnar_vs_mar(df, mask, X_cols, verbose=True):
    """
    Fit logistic regression R ~ X + Y and report coefficient on Y.
    If coef on Y is large (abs > 0.05 by default) selection is MNAR-like.
    Returns dict with coef_y.
    """
    R = mask.astype(int)
    X = df[X_cols].values.astype(float)
    y = df['BAD'].values.astype(float).reshape(-1, 1)
    mat = np.hstack([X, y])
    scaler = StandardScaler()
    mat_scaled = scaler.fit_transform(mat)
    # use a solver that handles no penalty
    try:
        model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
        model.fit(mat_scaled, R)
    except Exception:
        model = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=1000)
        model.fit(mat_scaled, R)
    coef = model.coef_.ravel()
    coef_y = coef[-1]
    if verbose:
        print(f"[diagnostic] logistic R ~ X + Y: coef on Y = {coef_y:.4f}  --> {'MNAR-like' if abs(coef_y) > 0.05 else 'MAR-like (small)'}")
        p_y1 = R[y.ravel()==1].mean() if (y.ravel()==1).any() else np.nan
        p_y0 = R[y.ravel()==0].mean() if (y.ravel()==0).any() else np.nan
        print(f"[diagnostic] P(R=1 | Y=1) = {p_y1:.3f}, P(R=1 | Y=0) = {p_y0:.3f}")
    return {'coef_y': float(coef_y)}

def compare_distributions(df, mask, X_cols):
    """Return summary differences between accepted and full population for X columns."""
    acc = df[mask]
    pop = df
    diffs = {}
    for c in X_cols:
        diffs[c] = {
            'pop_mean': float(pop[c].mean()),
            'acc_mean': float(acc[c].mean()),
            'pop_std': float(pop[c].std()),
            'acc_std': float(acc[c].std()),
        }
    return diffs

# convenience demo runner
def example_run(df, X_cols, seed=0):
    rng = np.random.default_rng(seed)
    print("Example MAR selection (logistic on X):")
    mask_mar, _ = selection_mar_logistic(df, X_cols, rng, intercept=0.0, stochastic=True)
    test_mnar_vs_mar(df, mask_mar, X_cols)
    print("\nExample MNAR selection (on Y):")
    mask_mnar, _ = selection_mnar_on_y(df, X_cols, rng, beta_y=2.0, intercept=-1.0, stochastic=True)
    test_mnar_vs_mar(df, mask_mnar, X_cols)
    print("\nLatent MNAR selection (unobserved U correlated with Y):")
    mask_latent, _ = selection_mnar_on_latent(df, X_cols, rng, gamma=1.5, latent_noise_scale=0.5, stochastic=True)
    test_mnar_vs_mar(df, mask_latent, X_cols)
    return