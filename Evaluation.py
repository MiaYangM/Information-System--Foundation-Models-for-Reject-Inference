import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from abc import ABC, abstractmethod

# Define a abstract class for all metrics will be used in evaluation methods
class Metric(ABC):
    '''
    Abstract base class for different evaluation metrics.
    '''
    @abstractmethod
    def __call__(self, 
                 y_true: np.ndarray, 
                 y_proba: np.ndarray, 
                 **kwargs) -> float: 
        pass

class AUC(Metric):
    '''
    Area Under the ROC Curve (AUC)
    '''
    def __call__(self, 
                 y_true: np.ndarray,
                 y_proba: np.ndarray, 
                 **kwargs) -> float:              
        return roc_auc_score(y_true=y_true, y_score=y_proba)

class BS(Metric):
    '''
    Brier Score (BS) metric.
    '''
    def __call__(self,  
                 y_true: np.ndarray, 
                 y_proba: np.ndarray, 
                 **kwargs) -> float:
        return brier_score_loss(y_true=y_true, y_prob=y_proba) # in 1.7.1, scikit learn updates y_prob to y_proba, that's why warnings exist.

class PAUC(Metric):
    '''
    Partial AUC over a specified False Negative Rate (FNR) range.
    '''
    def __call__(self, 
                 y_true: np.ndarray, 
                 y_proba: np.ndarray, 
                 **kwargs) -> float:
        fnr_range: list = kwargs.get('fnr_range', [0, 0.2])                     
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_proba)
        mask = ((tpr >= 1-fnr_range[1]) & (tpr <= 1-fnr_range[0]))
        if not np.any(mask):
            return 0.0
        pauc = np.trapz(tpr[mask], fpr[mask])
        return pauc
    
class ABR(Metric):
    '''
    Average Bad Rate (ABR) over a range of acceptance rates.
    '''
    def __call__(self, 
                 y_true: np.ndarray, 
                 y_proba: np.ndarray, 
                 **kwargs) -> float:
        acc_rate: list = kwargs.get('acc_rate', [0.2, 0.4])                   
        abrs = []
        alphas = np.linspace(acc_rate[0], acc_rate[1], num=10)
        y_true_array = np.asarray(y_true)
        for alpha in alphas:
            sorted_indices = np.argsort(y_proba)
            n_selected = int(len(y_proba) * alpha)
            selected = sorted_indices[:n_selected]
            y_acc = y_true_array[selected]
            abr = np.mean(y_acc)
            abrs.append(abr)
        return np.mean(abrs)
    
# Define a abstract class for all evaluation methods includes BE
class Evaluation(ABC):
    '''
    Abstract base class for evaluation frameworks.
    '''
    def __init__(self):
        pass

# Define BE as a subclass
class bayesianMetric(Evaluation):
    '''
    Performs Bayesian metric evaluation using Monte Carlo simulations for reject inference.

    Parameters:
    ----------
    metric: Metric
        A metric instance conforming to the Metric interface.
    '''
    def __init__(self, metric:Metric):
        self.metric = metric
        self.mean_vals_history_ = None
    
    def BM(self,
           y_true_acc: np.ndarray,
           y_proba_acc: np.ndarray,
           y_proba_rej: np.ndarray,
           rejects_prior: np.ndarray,
           acc_rate: list = [0.2, 0.4],
           seed: int = 312,
           min_iterations: int = 100,
           max_iterations: int = 10000,
           epsilon: float = 1e-4,
           fnr_range: list = [0, .2]) -> float:
        '''
        Perform Bayesian metric evaluation, and compute metric value of BE.

        Parameters:
        ----------
        y_true_acc : np.ndarray
            True labels for the accepts.
        y_proba_acc : np.ndarray
            Predicted probabilities for the accepts.
        y_proba_rej : np.ndarray
            Predicted probabilities for the rejects.
        rejects_prior : np.ndarray
            Prior probability array for the true labels of the rejects. Used to generate pseudo-labels during Monte Carlo simulations.
            Array length equals to sample size of rejects.
        acc_rate : List[float]
            A list or tuple specifying the range of acceptance rates (e.g., `[lower, upper]`).
            This parameter is passed to metric `ABR` if it is used..
        seed : int
            Random seed for reproducibility of the Monte Carlo simulations.
        min_iterations : int
            Minimum number of Monte Carlo iterations to perform before checking convergence. Defaults to 100.
        max_iterations : int
            Maximum number of Monte Carlo iterations to perform. The simulation will stop after this many iterations even if convergence is not met. Defaults to 10000.
        epsilon : float
            Convergence threshold. If the absolute difference between the mean of all metric values and the mean of values excluding the last one falls below this value, the iteration stops. Defaults to 1e-4.
        fnr_range : List[float]
            A list or tuple specifying the False Negative Rate (FNR) range (e.g., `[lower, upper]`).
            This parameter is passed to metric `PAUC` if it is used. Defaults to `[0, 0.2]`.

        Returns:
        -------
        float
            Estimated metric value for BE.
        '''
        np.random.seed(seed=seed)

        if isinstance(rejects_prior, (float, int)):
            rejects_prior = np.full(len(y_pred_rej), rejects_prior)

        # Merge predictions of accepts and rejects
        y_proba_combined = np.concatenate([y_proba_acc, y_proba_rej])
        vals = []
        mean_vals_history = []

        # Monte-Carlo simulation for pseudo-labels
        for n in range(max_iterations):
            y_sim_rej = np.random.binomial(n=1, p=rejects_prior)
            y_true_combined = np.concatenate([y_true_acc, y_sim_rej])

            # Compute metric
            val = self.metric(y_true_combined, y_proba_combined, fnr_range=fnr_range, acc_rate=acc_rate)
            vals.append(val)

            # Collect metric means of all previous iteration
            current_mean = np.mean(vals)
            mean_vals_history.append(current_mean)
            self.mean_vals_history_ = mean_vals_history

            # Check stopping criterion
            if n >= min_iterations:
                delta = abs(np.mean(vals[:-1]) - np.mean(vals))
                if delta < epsilon:
                    break

        return current_mean
    

if __name__ == "__main__":
    np.random.seed(2025)
    # Generate simulation data
    n_accepts = 1000
    n_rejects = 300
    y_true_acc = np.random.randint(0, 2, size=n_accepts)
    y_pred_acc = np.random.randint(0, 2, size=n_accepts)
    y_proba_acc = np.random.rand(n_accepts)

    y_pred_rej = np.random.randint(0, 2, size=n_rejects)
    y_proba_rej = np.random.rand(n_rejects)

    rejects_prior = np.random.rand(n_rejects)

    # Define metrics
    metric_auc = AUC()
    metric_bs = BS()
    metric_pauc = PAUC()
    metric_abr = ABR()

    # Calculate BM with different metrics
    bm_auc = bayesianMetric(metric_auc)
    bm_auc_value = bm_auc.BM(y_true_acc=y_true_acc,
                             y_proba_acc=y_proba_acc,
                             y_proba_rej=y_proba_rej,
                             rejects_prior=rejects_prior,
                             acc_rate=[0.2, 0.4],
                             fnr_range=[0, 0.2],
                             seed=2025)
    print(bm_auc_value)
    print(bm_auc.mean_vals_history_)
    
    bm_bs = bayesianMetric(metric_bs)
    bm_bs_value = bm_bs.BM(y_true_acc=y_true_acc,
                           y_proba_acc=y_proba_acc,
                           y_proba_rej=y_proba_rej,
                           rejects_prior=rejects_prior,
                           acc_rate=[0.2, 0.4],
                           fnr_range=[0, 0.2],
                           seed=2025)
    print(bm_bs_value)
    print(bm_bs.mean_vals_history_)

    bm_pauc = bayesianMetric(metric_pauc)
    bm_pauc_value = bm_pauc.BM(y_true_acc=y_true_acc,
                               y_proba_acc=y_proba_acc,
                               y_proba_rej=y_proba_rej,
                               rejects_prior=rejects_prior,
                               acc_rate=[0.2, 0.4],
                               fnr_range=[0, 0.2],
                               seed=2025)
    print(bm_pauc_value)
    print(bm_pauc.mean_vals_history_)

    bm_abr = bayesianMetric(metric_abr)
    bm_abr_value = bm_abr.BM(y_true_acc=y_true_acc,
                             y_proba_acc=y_proba_acc,
                             y_proba_rej=y_proba_rej,
                             rejects_prior=rejects_prior,
                             acc_rate=[0.2, 0.4],
                             fnr_range=[0, 0.2],
                             seed=2025)
    print(bm_abr_value)
    print(bm_abr.mean_vals_history_)
