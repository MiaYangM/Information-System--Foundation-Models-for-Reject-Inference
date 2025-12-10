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
                 bad_ratio = 0.5,
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
            'bin_prob': 0.5, 'bin_mean_bad_dif': 0, 'bin_bad_ratio': 0.5,
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

    def args_update(self):
        if self.replicate is not None and hasattr(self.replicate, "args"):
            for key, value in self.replicate.args.items():
                setattr(self, key, value)

        for key, value in self.user_params.items():
            setattr(self, key, value)

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