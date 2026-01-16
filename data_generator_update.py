import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid


class DataGenerator2:
    def __init__(self,
                 # number of samples, and variables
                 n = 1000,
                 k_con = 10,
                 k_bin = 2,
                 # ratio
                 bad_ratio = 0.8,
                 # continuous variables (these influence feature generation and can be used for default coefs)
                 con_mean_bad_dif = 1,
                 con_nonlinear = 0.5,
                 con_noise_var = 0.1,
                 con_var_bad_dif = 0,
                 covars = None,
                 # binary variables
                 bin_prob = 0.5,
                 bin_mean_bad_dif = 0,
                 bin_bad_ratio = 0.5,
                 bin_mean_con_dif = 0,
                 bin_var_bad_dif = 0,
                 bin_noise_var = 0.1,
                 # noise / signal controls (new)
                 coef_scale = 1.0,     # multiply the label-generation coefficients (reduce to make task harder)
                 noise_scale = 1.0,    # divide logits by this value to make probabilities closer to 0.5
                 flip_frac = 0.0,      # fraction of labels to randomly flip to introduce label noise
                 # rest
                 verbose = True,
                 seed = None,
                 replicate = None
                 ):

        default_params = {
            'n': 1000, 'k_con': 10, 'k_bin': 2, 'bad_ratio': 0.5,
            'con_mean_bad_dif': 1, 'con_nonlinear': 0.5, 'con_noise_var': 0.1,
            'con_var_bad_dif': 0, 'covars': None,
            'bin_prob': 0.5, 'bin_mean_bad_dif': 0, 'bin_bad_ratio': 0.8,
            'bin_mean_con_dif': 0, 'bin_var_bad_dif': 0, 'bin_noise_var': 0.1,
            'coef_scale': 1.0, 'noise_scale': 1.0, 'flip_frac': 0.0,
            'verbose': True, 'seed': None, 'replicate': None, 'data': []
        }

        # capture user-overridden params as before
        call_args = locals()
        self.user_params = {k: v for k, v in call_args.items()
                            if k != 'self' and k in default_params and v != default_params[k]}


        # core settings
        self.n = int(n)
        self.k_con = int(k_con)
        self.k_bin = int(k_bin)
        self.bad_ratio = float(bad_ratio)
        self.con_mean_bad_dif = con_mean_bad_dif
        self.con_nonlinear = con_nonlinear
        self.con_noise_var = con_noise_var
        self.con_var_bad_dif = con_var_bad_dif
        self.covars = covars

        # binary feature settings
        self.bin_prob = bin_prob
        self.bin_mean_bad_dif = bin_mean_bad_dif
        self.bin_bad_ratio = bin_bad_ratio
        self.bin_mean_con_dif = bin_mean_con_dif
        self.bin_var_bad_dif = bin_var_bad_dif
        self.bin_noise_var = bin_noise_var

        # new signal/noise params
        self.coef_scale = float(coef_scale)
        self.noise_scale = float(noise_scale)
        self.flip_frac = float(flip_frac)

        self.verbose = verbose
        self.seed = None if seed is None else int(seed)
        self.replicate = replicate

        # internal storage
        self.data = None
        self._rng = np.random.RandomState(self.seed)
        self._coefs = None
        self.con_params = {
            'means': [],
            'covar': [],
            'combo': []
        }
        self.args = {}
        
       
        self.user_params = {k: v for k, v in call_args.items()
                    if k != 'self' and k not in ['replicate', 'default_params', 'call_args']
                    and v != default_params.get(k)}

        print(f"[DEBUG - Before args_update] Initial user_params: {self.user_params}")
        self.args_update()
        print(f"[DEBUG - After args_update] Updated user_params: {self.user_params}")

 
    


    # --- replace your existing args_update with this safer version ---
    def args_update(self):
        # Copy (whitelist) attributes from replicate if provided,
        # but do NOT copy internal/nested objects or overwrite explicit user overrides.
        if self.replicate is not None:
            print("[DEBUG] Replication enabled. Copying attributes from replicate...")
            for key, value in vars(self.replicate).items():
                # Skip private / internal or complex nested attributes
                if key.startswith("_"):
                    continue
                if key in ("user_params", "call_args", "default_params", "data", "args", "con_params"):
                    continue
                # Only set attribute if the current instance did not explicitly override it
                if hasattr(self, key) and key not in self.user_params:
                    setattr(self, key, value)
                    print(f"[DEBUG] Attribute '{key}' replicated with value: {value}")
                else:
                    if hasattr(self, key):
                        print(f"[DEBUG] Attribute '{key}' skipped due to user override ({self.user_params.get(key)})")

        # Finally, apply explicitly user-provided parameters (they take precedence)
        print("[DEBUG] Applying user-provided parameters...")
        for key, value in self.user_params.items():
            setattr(self, key, value)
            print(f"[DEBUG] Overwriting/Setting user-param '{key}' to: {value}")

        # Refresh user_params to reflect the final state (only keep keys that were explicitly provided)
        self.user_params = {k: getattr(self, k) for k in self.user_params.keys()}
        print(f"[DEBUG] Updated user_params: {self.user_params}")
# ------------------------------------------------------------------------

        # Apply explicitly user-provided parameters to override replicated values
        print("[DEBUG] Applying user-provided parameters...")
        for key, value in self.user_params.items():
            setattr(self, key, value)
            print(f"[DEBUG] Overwriting/Setting user-param '{key}' to: {value}")

        # Refresh user_params to reflect the current instance's state
        self.user_params = {key: getattr(self, key) for key in self.user_params.keys()}
        print(f"[DEBUG] Updated user_params: {self.user_params}")

        # Refresh user_params with updated instance values
        self.user_params = {k: getattr(self, k) for k in self.user_params.keys()}
        print(f"[DEBUG] Updated user_params: {self.user_params}")

    def args_summary(self):
        if self.replicate is not None:
            self.args_update()

        if self.k_con < 1:
            raise ValueError("At least one continuous feature is required")
        if self.k_bin < 0:
            raise ValueError("No negative binary features are allowed")

        k = self.k_con + self.k_bin
        if self.verbose:
            print('Generating {} continuous features with {} binary features'.format(self.k_con, self.k_bin))
            print('Simulating ({} x {}) data set'.format(self.n, k))

    def generate_pos_def_matrix(self):
        # Generate symmetric random matrix and make PD
        A = self._rng.uniform(0, 1, (self.k_con, self.k_con))
        Sigma = np.dot(A, A.T)
        Sigma += np.eye(self.k_con) * 1e-6
        return Sigma

    def _default_coefs(self):
        # Create a default coefficient vector from con_mean_bad_dif if provided
        if isinstance(self.con_mean_bad_dif, (list, tuple, np.ndarray)):
            arr = np.array(self.con_mean_bad_dif, dtype=float)
            if arr.size >= self.k_con:
                return arr[:self.k_con]
            else:
                # pad or repeat
                rep = np.ones(self.k_con)
                rep[:arr.size] = arr
                return rep
        else:
            # scalar difference -> spread that across features
            return np.ones(self.k_con, dtype=float) * float(self.con_mean_bad_dif)

    def generate(self, shuffle=True):
        """
        Generate synthetic dataset and store in self.data (DataFrame).
        - continuous features X1..Xk (Gaussian)
        - binary features B1..Bk_bin (Bernoulli)
        - label BAD/GOOD generated from logistic model controlled by coef_scale/noise_scale/flip_frac
        """
        self.args_summary()

        rng = self._rng

        # 1) simulate continuous features X ~ N(0, I) by default (or use covars if provided)
        if self.covars is None:
            X = rng.normal(loc=0.0, scale=1.0, size=(self.n, self.k_con))
        else:
            # if covars provided, attempt to use them as covariance for multivariate normal
            try:
                Sigma = np.asarray(self.covars)
                if Sigma.shape == (self.k_con, self.k_con):
                    X = rng.multivariate_normal(np.zeros(self.k_con), Sigma, size=self.n)
                else:
                    X = rng.normal(loc=0.0, scale=1.0, size=(self.n, self.k_con))
            except Exception:
                X = rng.normal(loc=0.0, scale=1.0, size=(self.n, self.k_con))

        # 2) simulate binary features if requested
        B = None
        if self.k_bin > 0:
            B = rng.binomial(1, self.bin_prob, size=(self.n, self.k_bin))

        # 3) construct a linear score and probability for BAD using default or random coefs
        base_coefs = self._default_coefs().astype(float)
        # if base_coefs length differs from k_con, pad/truncate
        if base_coefs.size < self.k_con:
            coefs = np.pad(base_coefs, (0, self.k_con - base_coefs.size), constant_values=0.0)
        else:
            coefs = base_coefs[:self.k_con].copy()

        # store original coefs for external inspection
        self._coefs = coefs.copy()

        # apply coef_scale (weaken/strengthen signal)
        coefs = coefs * float(self.coef_scale)

        # compute raw linear combination
        linear_score = X.dot(coefs)

        # 4) find intercept so that mean_p_bad ~= bad_ratio (simple bisection)
        # compute noisy logits = linear_score / noise_scale + intercept
        def mean_with_intercept(intercept):
            logits = (linear_score + intercept) / max(1e-12, self.noise_scale)
            return float(np.mean(sigmoid(logits)))

        # target prevalence
        target = float(self.bad_ratio)
        print(f"[DEBUG] Set bad_ratio: {self.bad_ratio}") 
        # simple bracket search for intercept
        lo, hi = -50.0, 50.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            m = mean_with_intercept(mid)
            if abs(m - target) < 1e-6:
                intercept = mid
                break
            if m < target:
                lo = mid
            else:
                hi = mid
        else:
            intercept = mid

        # final probabilities and sampling
        logits = (linear_score + intercept) / max(1e-12, self.noise_scale)
        p_bad = sigmoid(logits)
        y = (rng.rand(self.n) < p_bad).astype(int)

        # 5) apply label flips if requested (introduce label noise)
        if self.flip_frac and self.flip_frac > 0.0:
            n_flip = int(round(self.n * float(self.flip_frac)))
            flip_idx = rng.choice(self.n, size=n_flip, replace=False)
            y[flip_idx] = 1 - y[flip_idx]

        # 6) assemble DataFrame
        cols = {}
        for i in range(self.k_con):
            cols[f"X{i+1}"] = X[:, i]
        # add BAD column as 'BAD'/'GOOD' strings (keeps compatibility with existing notebook)
        cols["BAD"] = np.where(y == 1, "BAD", "GOOD")

        # binary columns B1..Bk
        if self.k_bin > 0:
            for j in range(self.k_bin):
                cols[f"B{j+1}"] = B[:, j]

        df = pd.DataFrame(cols)

        # optional shuffle so rows are not ordered by anything
        if shuffle:
            df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        self.data = df
        # keep numeric y and p_bad for external use
        self._y_numeric = y
        self._p_bad = p_bad
        self._intercept = intercept

        if self.verbose:
            print(f"Generated dataset with n={self.n}, k_con={self.k_con}, k_bin={self.k_bin}")
            print(f"BAD prevalence (approx): {np.mean(y):.4f} (target {self.bad_ratio})")
            print(f"coef_scale={self.coef_scale}, noise_scale={self.noise_scale}, flip_frac={self.flip_frac}")

        return df
 

    

    
    def impose_selection(self,
                     mode='MNAR',
                     top_percent=None,
                     beta_x=None,
                     beta_y=1.0,
                     alpha=1.0,
                     target_accept_rate=None,
                     probabilistic=True,
                     stochastic=True,
                     rng_seed=None):
        """
        Impose MAR/MNAR/MIXED selection on self.data and return (df_full, df_obs).
        Place this method inside DataGenerator2 class.
        """
        if self.data is None:
            raise ValueError("No data generated: call generate() first")

    

        rng = np.random.default_rng(self.seed if rng_seed is None else rng_seed)
        df_full = self.data.copy().reset_index(drop=True)

        # numeric Y column 0/1
        if 'BAD' not in df_full.columns:
            raise ValueError("Data missing 'BAD' column")
        df_full['Y'] = df_full['BAD'].map({'GOOD':0, 'BAD':1}).astype(int)

        # pick X columns (continuous features named X1..Xk by generator convention)
        X_cols = [c for c in df_full.columns if c.startswith('X')]
        if len(X_cols) == 0:
            raise ValueError("No feature columns found (expected columns starting with 'X')")
        X = df_full[X_cols].astype(float).values

        # default beta_x if None
        if beta_x is None:
            rng_local = np.random.RandomState(int(self.seed or 0))
            beta_x = rng_local.normal(scale=1.0, size=X.shape[1])

        score_x = X.dot(np.asarray(beta_x))
        score_y = beta_y * df_full['Y'].values.astype(float)

        if mode == 'MAR':
            score = score_x.copy()
        elif mode == 'MNAR':
            score = score_x + score_y
        elif mode == 'MIXED':
            score = (1.0 - alpha) * score_x + alpha * score_y
        else:
            raise ValueError("mode must be 'MAR','MNAR' or 'MIXED'")

        # deterministic top-percent selection
        if top_percent is not None:
            thr = np.quantile(score, 1 - float(top_percent))
            mask_accept = score >= thr
            p_accept = None
        else:
            # probabilistic selection
            logits = score - np.mean(score)
            intercept = 0.0
            if target_accept_rate is not None:
                lo, hi = -50.0, 50.0
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    p = sigmoid(logits + mid)
                    m = float(p.mean())
                    if abs(m - float(target_accept_rate)) < 1e-6:
                        intercept = mid
                        break
                    if m < target_accept_rate:
                        lo = mid
                    else:
                        hi = mid
                else:
                    intercept = mid
            prob = sigmoid(logits + intercept)
            if stochastic:
                mask_accept = rng.random(len(prob)) < prob
            else:
                # default to top 20% if no top_percent provided
                thr = np.quantile(prob, 1 - (top_percent or 0.2))
                mask_accept = prob >= thr
            p_accept = prob

        df_obs = df_full.copy()
        df_obs['S'] = 0
        df_obs.loc[mask_accept, 'S'] = 1
        df_obs['Y_obs'] = df_obs['Y'].where(df_obs['S'] == 1, np.nan)
        df_obs['p_accept'] = p_accept if p_accept is not None else df_obs['S'].astype(float)

        return df_full, df_obs