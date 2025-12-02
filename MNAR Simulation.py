# Example helpers to add to your existing data generator
import numpy as np
import pandas as pd
from scipy.special import expit  # logistic

def apply_selection(df, mode='MAR', top_percent=0.2, gamma=1.0, latent_noise=1.0, seed=None, probabilistic=True):
    """
    df: DataFrame with features and ground-truth 'BAD' column as 0/1
    mode:
      - 'MAR' : selection by observed features only (e.g., X1 quantile) -- existing behavior
      - 'MNAR': selection depends on true label (Y) or latent correlated with Y
    gamma: strength of dependence on Y (positive -> more likely to ACCEPT GOOD if Y==0)
    probabilistic: if True use logistic probability, else deterministic threshold on score
    Returns: df_full (oracle), df_observed (with 'S' and Y masked where S==0)
    """
    rng = np.random.RandomState(seed)
    df_full = df.copy().reset_index(drop=True)
    # ensure numeric Y (0=GOOD,1=BAD)
    if df_full['BAD'].dtype != np.number:
        df_full['Y'] = df_full['BAD'].map({'GOOD':0, 'BAD':1})
    else:
        df_full['Y'] = df_full['BAD'].astype(int)

    # Base acceptance score from observed features (same as MAR)
    base_score = df_full['X1'].values.astype(float)

    if mode == 'MAR':
        # deterministic: accept top percentile by observed X1 (existing)
        thr = np.quantile(base_score, 1 - top_percent)
        mask_accept = base_score >= thr

    elif mode == 'MNAR':
        # create a latent that correlates with Y (optional)
        latent = df_full['Y'].values + rng.normal(scale=latent_noise, size=len(df_full))

        # combine observed base_score with latent/true Y effect
        # gamma controls how strongly Y/latent shifts acceptance odds
        selection_logit = 0.5 * (base_score - np.median(base_score)) + gamma * (1.0 - df_full['Y'].values)
        #  (example: GOOD (Y=0) increases logit if gamma>0, making GOOD more likely to be accepted)

        if probabilistic:
            p_accept = expit(selection_logit)  # in (0,1)
            mask_accept = rng.rand(len(df_full)) < p_accept
        else:
            # deterministic: choose top fraction of selection_logit
            thr = np.quantile(selection_logit, 1 - top_percent)
            mask_accept = selection_logit >= thr

    else:
        raise ValueError("mode must be 'MAR' or 'MNAR'")

    df_obs = df_full.copy()
    df_obs['S'] = 0
    df_obs.loc[mask_accept, 'S'] = 1

    # Observed Y only for accepted rows; set to NaN for rejected to simulate missingness
    df_obs['Y_obs'] = df_obs['Y'].where(df_obs['S'] == 1, np.nan)

    return df_full, df_obs