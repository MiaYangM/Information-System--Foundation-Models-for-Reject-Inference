import numpy as np
import pandas as pd
import json

class DataGenerator:
    def __init__(self,
                 # number of samples, and variables
                 n = 1000,
                 k_con = 10,
                 k_bin = 2,
                 # ratio
                 bad_ratio = 0.5,
                 # continuous variables
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
            'verbose': True, 'seed': None, 'replicate': None, 'data': []
        }
        
        # Get all parameters from the function call
        call_args = locals()
        
        # Only keep parameters that differ from defaults
        self.user_params = {k: v for k, v in call_args.items() 
                        if k != 'self' and k in default_params and v != default_params[k]}

        self.n = n
        self.k_con = k_con
        self.k_bin = k_bin
        self.bad_ratio = bad_ratio
        self.con_mean_bad_dif = con_mean_bad_dif
        self.con_nonlinear = con_nonlinear
        self.con_noise_var = con_noise_var
        self.con_var_bad_dif = con_var_bad_dif
        self.covars = covars
        self.bin_prob = bin_prob
        self.bin_mean_bad_dif = bin_mean_bad_dif
        self.bin_bad_ratio = bin_bad_ratio
        self.bin_mean_con_dif = bin_mean_con_dif
        self.bin_var_bad_dif = bin_var_bad_dif
        self.bin_noise_var = bin_noise_var
        self.verbose = verbose
        self.seed = seed
        self.replicate = replicate

        self.data = []
        self.con_params = {
            'means': [], 
            'covar': [],
            'combo': []
        }
        self.args = {}
            

    def args_update(self):
        for key, value in self.replicate.args.items():
            setattr(self, key, value)

        for key, value in self.user_params.items():
            setattr(self, key, value)


    def args_summary(self):
        if self.replicate is not None:
            self.args_update()
            pass

        if self.k_con < 1:
            raise ValueError("At least one continuous feature is required")

        if self.k_bin < 0:
            raise ValueError("No negative binary features are allowed")
        
        k = self.k_con + self.k_bin
        print('Generating {} continuous features with {} binary features'.format(self.k_con, self.k_bin))
        print('Simulating ({} x {}) data set'.format(self.n, k))


    def generate_pos_def_matrix(self):
        # Generate symmetric random matrix
        A = np.random.uniform(0, 1, (self.k_con, self.k_con))
        Sigma = np.dot(A, A.T)
        # to make it positive definite
        Sigma += np.eye(self.k_con) * 1e-6

        return Sigma

    def generate_cov(self, combo_vals):

        sigma1, sigma2 = self.generate_pos_def_matrix(), self.generate_pos_def_matrix()

        if np.sum(combo_vals) > 0:
            combo_var_dif = np.mean(np.array(self.bin_var_bad_dif) * combo_vals)
        else:
            combo_var_dif = self.con_var_bad_dif

        # Randomly decide which elements to replace
        num_off_diagonal = int((self.k_con**2 - self.k_con) / 2)
        sigma_index = np.random.binomial(1, combo_var_dif, num_off_diagonal).astype(bool)

        # Indices for upper/lower triangle
        upper_indices = np.triu_indices(self.k_con, k=1)
        lower_indices = np.tril_indices(self.k_con, k=-1)

        # Replace upper triangle
        sigma2[upper_indices[0][~sigma_index], upper_indices[1][~sigma_index]] = sigma1[upper_indices[0][~sigma_index], upper_indices[1][~sigma_index]]
        # Replace lower triangle
        sigma2[lower_indices[0][~sigma_index], lower_indices[1][~sigma_index]] = sigma1[lower_indices[0][~sigma_index], lower_indices[1][~sigma_index]]

        # Replace diagonal
        sigma_index_diag = np.random.binomial(1, combo_var_dif, self.k_con).astype(bool)
        sigma2[np.arange(self.k_con)[~sigma_index_diag], np.arange(self.k_con)[~sigma_index_diag]] = sigma1[np.arange(self.k_con)[~sigma_index_diag], np.arange(self.k_con)[~sigma_index_diag]]

        # Ensure positive semi-definite
        ev = np.linalg.eigvalsh(sigma2)
        while not np.all(ev >= -np.sqrt(np.finfo(float).eps) * np.abs(ev[0])):
            sigma2 = self.generate_pos_def_matrix()
            sigma_index = np.random.binomial(1, combo_var_dif, num_off_diagonal).astype(bool)
            
            sigma2[upper_indices[0][~sigma_index], upper_indices[1][~sigma_index]] = sigma1[upper_indices[0][~sigma_index], upper_indices[1][~sigma_index]]
            sigma2[lower_indices[0][~sigma_index], lower_indices[1][~sigma_index]] = sigma1[lower_indices[0][~sigma_index], lower_indices[1][~sigma_index]]
            sigma_index_diag = np.random.binomial(1, combo_var_dif, self.k_con).astype(bool)
            sigma2[np.arange(self.k_con)[~sigma_index_diag], np.arange(self.k_con)[~sigma_index_diag]] = sigma1[np.arange(self.k_con)[~sigma_index_diag], np.arange(self.k_con)[~sigma_index_diag]]
            ev = np.linalg.eigvalsh(sigma2)

        return sigma1, sigma2


    def generate_binary(self):
        print('Generating binary features...')
        binary_df = []
        
        if self.k_bin == 0:
            binary_df = pd.DataFrame({'B1': np.ones(self.n, dtype=int)})
            combos = ['1']
        else:
            bin_prob = [self.bin_prob]*self.k_bin
            # generate data
            for i in range(self.k_bin):
                # print(i)
                binary_df.append(np.random.binomial(n = 1, size = self.n, p = bin_prob[i]))    
            binary_df = np.column_stack(binary_df)

            # noise injection for binary features
            for i in range(binary_df.shape[1]):
                noise = np.random.binomial(1, self.bin_noise_var, self.n)
                binary_df[:, i] = np.abs(binary_df[:, i] - noise)

            combos = np.unique([''.join(map(str, row)) for row in binary_df])
            binary_df = pd.DataFrame(binary_df, columns=[f'B{i+1}' for i in range(self.k_bin)])
        
        return binary_df, combos

    def generate_continuous(self, binary_df, combos):
        for combo in combos:
            idx = np.where(np.array(combos) == combo)[0][0]
            combo_vals = np.array([int(x) for x in combo])
            combo_idx = np.all(binary_df == combo_vals, axis=1)
            combo_count = np.sum(combo_idx)

            # Bad ratio assign
            if np.sum(combo_vals) > 0 and self.k_bin != 0:
                combo_bad_ratio = np.mean(np.array(self.bin_bad_ratio) * combo_vals)
            else:
                combo_bad_ratio = self.bad_ratio
         
            # Compute sample sizes
            combo_n1 = int(round(combo_count * combo_bad_ratio))
            combo_n2 = int(round(combo_count * (1 - combo_bad_ratio)))
            if combo_n1 + combo_n2 != combo_count:
                cid = np.random.binomial(1, 0.5)
                if cid == 0:
                    combo_n1 = combo_count - combo_n2
                else:
                    combo_n2 = combo_count - combo_n1
            
            if self.replicate is None:
                mu1 = np.repeat(np.sum(combo_vals * self.bin_mean_con_dif), self.k_con)
                mu2 = mu1 + self.con_mean_bad_dif + np.sum(combo_vals * self.bin_mean_bad_dif)
            else:
                mu1 = self.replicate.con_params['means'][idx][0, :] 
                mu2 = self.replicate.con_params['means'][idx][1, :]           

            if self.covars:
                sigma1 = np.array(self.covars[0]).reshape(2,2)
                sigma2 = np.array(self.covars[1]).reshape(2,2) 
            else:
                sigma1, sigma2 = self.generate_cov(combo_vals)

            self.con_params['means'].append(np.vstack((mu1, mu2)))
            self.con_params['covar'].append([sigma1, sigma2])
            self.con_params['combo'].append(combo_vals)

            rng = np.random.default_rng(self.seed)
            X1 = None
            X2 = None

            if combo_n1 > 0:
                X1 = pd.DataFrame(rng.multivariate_normal(mean=mu1, cov=sigma1, size=combo_n1))
                X1['BAD'] = 'BAD'

            if combo_n2 > 0:
                X2 = pd.DataFrame(rng.multivariate_normal(mean=mu2, cov=sigma2, size=combo_n2))
                X2['BAD'] = 'GOOD'

            X = pd.concat([X1, X2], ignore_index=True)
            X['BAD'] = X['BAD'].astype('category')

            combo_index = np.where(combo_idx)[0]
            if list(combos).index(combo) == 0:
                self.data = pd.concat([X, binary_df.loc[combo_index].reset_index(drop=True)], axis=1)
            else:
                for i in range(combo_vals.shape[0]):
                    X[f'B{i+1}'] = combo_vals[i]
                
                self.data = pd.concat([self.data, X], ignore_index=True)
        
        for i, col in enumerate(self.data.columns[:self.k_con], start=1):
            self.data.rename(columns={col: f'X{i}'}, inplace=True)
            

    def inject_noise_nonLinear(self):
        if self.con_nonlinear > 0 and round(self.con_nonlinear * self.k_con) > 0:
            nonlinear_vars = np.random.choice(self.k_con, size=round(self.con_nonlinear * self.k_con), replace=False)
            nonlinear_vars = {col: f'X_{col}' for col in nonlinear_vars}
            nonlinear_type = np.random.binomial(1, 0.5, size=len(nonlinear_vars)) + 1
            nonlinear_desc = ['square' if t == 1 else 'cube' for t in nonlinear_type]
            nonlinearities = ', '.join([f"{var}: {desc}" for var, desc in zip(nonlinear_vars, nonlinear_desc)])

            for i, var in enumerate(nonlinear_vars):
                if nonlinear_type[i] == 1:
                    self.data.iloc[:, int(var)] = self.data.iloc[:, int(var)]**2
                elif nonlinear_type[i] == 2:
                    self.data.iloc[:, int(var)] = self.data.iloc[:, int(var)]**3
        else:
            nonlinearities = None

        #Inject noise
        for i in range(self.k_con):
            self.data.iloc[:, i] = self.data.iloc[:, i] + np.random.normal(0, self.con_noise_var, size=self.n)


    def save_args(self):
        args = {
            # Dimensionality parameters
            'n': self.n,
            'k_con': self.k_con,
            'k_bin': self.k_bin,
            'bad_ratio': self.bad_ratio,
            
            # Continuous features parameters
            'con_mean_bad_dif': self.con_mean_bad_dif,
            'con_var_bad_dif': self.con_var_bad_dif,
            'con_nonlinear': self.con_nonlinear,
            'con_noise_var': self.con_noise_var,
            'covars': self.covars,            
            # Binary features parameters
            'bin_prob': self.bin_prob,
            'bin_mean_bad_dif': self.bin_mean_bad_dif,
            'bad_ratios': self.bin_bad_ratio,
            'bin_mean_con_dif': self.bin_mean_con_dif,
            'bin_var_bad_dif': self.bin_var_bad_dif,
            'bin_noise_var': self.bin_noise_var,
            
            # Other parameters
            'seed': self.seed,
            'verbose': self.verbose
        }
        self. args = args

    def generate(self):
        if self.replicate is not None:
            self.args_update()

        self.args_summary()
        binary_df, combos = self.generate_binary()
        self.generate_continuous(binary_df, combos)
        self.inject_noise_nonLinear()
        self.save_args()