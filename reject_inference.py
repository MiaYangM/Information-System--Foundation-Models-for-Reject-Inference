# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError # Import NotFittedError
from xgboost import XGBClassifier 
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize # Import minimize
import statsmodels.api as sm # Import statsmodels.api

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict
from Evaluation import Metric, AUC, PAUC, BS, ABR, Evaluation, bayesianMetric

import random
import copy
import warnings

# Define a small epsilon for numerical stability
EPS = 1e-10

class RejectInference(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, accepts_x: pd.DataFrame, accepts_y: pd.Series, rejects_x: pd.DataFrame, holdout_x: pd.DataFrame):
        """
        Abstract method to build the reject inference model.

        Parameters
        ----------
        accepts_x : pd.DataFrame
            Features of the accepted (observed) samples.
        accepts_y : pd.Series
            True labels of the accepted samples.
        rejects_x : pd.DataFrame
            Features of the rejected samples whose outcomes are unknown.
            This is the *original full set* of rejected data.
        holdout_x : pd.DataFrame
            Features of the holdout sample, for evaluation purposes.

        Returns
        -------
        self : object
            Returns the instance itself after fitting.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X using the fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Features of the samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels (0 or 1).

        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Abstract method to predict class probabilities for samples in X.

        Subclasses must implement this method to provide probability
        predictions using their internal fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities. For binary classification,
            this is typically an array with two columns (probability of class 0,
            probability of class 1).

        """
        pass

class BiasAwareSelfLearning(RejectInference):

    def __init__(self, strong_estimator: object, weak_learner_estimator: object,
                 max_iterations: int = 3,
                 silent: bool = False,
                 filtering_beta: tuple = (0.05, 1.0),
                 sampling_percent: float = 0.8,
                 holdout_percent: float = 0.1,
                 labeling_percent=0.01,
                 multiplier=2.0,
                 early_stop: bool = False,
                 evaluation_metric: Metric = AUC(),
                 bayesian_acc_rate: List[float] = [0.2, 0.4], # Add parameter for acc_rate used in Bayesian Metric
                 bayesian_fnr_range: List[float] = [0, 0.2]): # Add parameter for fnr_range used in Bayesian Metric
        """
        Parameters
        ----------
        strong_estimator : object
            The 'strong' base estimator for final model training.
        weak_learner_estimator : object
            The 'weak' base estimator for iterative pseudo-labeling.
        max_iterations : int, default=3
            Maximum number of self-learning iterations.
        silent : bool, default=False
            Whether to print training details.
        filtering_beta : tuple, default=(0.05, 1.0)
            Percentiles of rejects to be filtered (e.g., (0.1, 0.9) to keep middle 80%).
        sampling_percent : float, default=0.8
            Percentage of remaining rejects to be sampled in each labeling iteration. ρ in Algorithm B.2
        labeling_percent : float, default=0.01
            Percentage of sampled rejects to be labeled (for the 'bad' class). 	γ in Algorithm B.2
            This defines the upper quantile for 'bad' predictions.
        multiplier : float, default=2. θ in Algorithm B.2
            Imbalance multiplier for 'bad' vs 'good' rejects during labeling.
            `per_g = labeling_percent / multiplier` means 'good' labels are `1/multiplier` times less likely to be selected than 'bad' based on quantile.
            We expect the bad rate in reject to be higher than in accept.Therefore,detting 𝜃 >1 helps to append
            more bad than good examples, increasing the bad rate in the training sample to approximate the population distribution.
        early_stop : bool, default=False
            Whether to use performance-based early stopping. If True, the labeling
            iterations will stop if the performance metric on the holdout set
            does not improve.
        evaluation_metric : Metric, default=AUC()
            An instance of a Metric (e.g., AUC, BS, PAUC, ABR) to be used by the
            Bayesian evaluation (BM).
        bayesian_acc_rate : List[float], default=[0.2, 0.4]
            Acceptance rate range for the Bayesian Metric evaluation (used by ABR).
        bayesian_fnr_range : List[float], default=[0, 0.2]
            False Negative Rate range for the Bayesian Metric evaluation (used by PAUC).
        """
        # Call parent constructor first
        super().__init__()

        # Store parameters specific to BiasAwareSelfLearning
        self.strong_estimator = strong_estimator # Stored as strong_estimator
        self.weak_learner_estimator = weak_learner_estimator
        self.max_iterations = max_iterations
        self.silent = silent
        self.filtering_beta = filtering_beta
        self.sampling_percent = sampling_percent
        self.holdout_percent = holdout_percent
        self.labeling_percent = labeling_percent
        self.multiplier = multiplier
        self.early_stop = early_stop
        self.evaluation_metric = evaluation_metric
        self.bayesian_acc_rate = bayesian_acc_rate # Store the bayesian_acc_rate
        self.bayesian_fnr_range = bayesian_fnr_range # Store the bayesian_fnr_range


        # Initialize the Bayesian evaluator instance. It will use its own defaults for BM params.
        # The BM method itself accepts acc_rate and fnr_range, which is handled when BM is called.
        self.bayesian_evaluator = bayesianMetric(metric=self.evaluation_metric)

        # --- Parameter Validations ---

        # Validate strong_estimator
        if self.strong_estimator is None:
            raise ValueError("`strong_estimator` cannot be None.")
        if not hasattr(self.strong_estimator, 'predict_proba'):
            raise AttributeError("Strong learner (`strong_estimator`) must support `predict_proba` method.")

        # Validate weak_learner_estimator
        if self.weak_learner_estimator is None:
            raise ValueError("`weak_learner_estimator` cannot be None.")

        # Validate labeling_percent
        if not isinstance(self.labeling_percent, (int, float)) or not (0 < self.labeling_percent < 1):
            raise ValueError("`labeling_percent` must be a float strictly between 0 and 1.")

        # Validation for multiplier
        if not (self.multiplier > 1):
            raise ValueError("`multiplier` should be a float strictly greater than 1. "
                             "We expect the bad rate in reject to be higher than in accept.")

        # Validate max_iterations
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("`max_iterations` must be an integer greater than or equal to 1.")

        # Validate silent
        if not isinstance(self.silent, bool):
            raise ValueError("`silent` must be a boolean.")

        # Validate filtering_beta
        if not isinstance(self.filtering_beta, (tuple, list)) or len(self.filtering_beta) != 2:
            raise ValueError("`filtering_beta` must be a tuple or list of two floats.")
        if not all(isinstance(val, (int, float)) for val in self.filtering_beta):
            raise ValueError("Elements of `filtering_beta` must be floats or integers.")
        if not (0.0 <= self.filtering_beta[0] <= 1.0 and 0.0 <= self.filtering_beta[1] <= 1.0):
            raise ValueError("Elements of `filtering_beta` must be between 0.0 and 1.0 (inclusive).")
        if not (self.filtering_beta[0] <= self.filtering_beta[1]):
            raise ValueError("The first element of `filtering_beta` (lower bound) must be less than or equal to the second (upper bound).")

        # Validate sampling_percent
        if not isinstance(self.sampling_percent, (int, float)) or not (0 < self.sampling_percent < 1):
            raise ValueError("`sampling_percent` must be a float strictly between 0 and 1.")

        # Validate holdout_percent
        if not isinstance(self.holdout_percent, (int, float)) or not (0 <= self.holdout_percent < 1):
            raise ValueError("`holdout_percent` must be a float between 0 (inclusive) and 1 (exclusive).")

        # Validate holdout_percent
        if not isinstance(self.holdout_percent, (int, float)) or not (0 <= self.holdout_percent < 1): # Changed to <1, as 1 would mean no training data.
            raise ValueError("`holdout_percent` must be a float between 0 (inclusive) and 1 (exclusive).")

        # Validate early_stop
        if not isinstance(self.early_stop, bool):
            raise ValueError("`early_stop` must be a boolean.")

        # Validate evaluation_metric
        if not isinstance(self.evaluation_metric, Metric):
            raise TypeError("`evaluation_metric` must be an instance of a `Metric` subclass (e.g., AUC, BS, PAUC, ABR).")



    #helper function for filtering, reference to Algorithm B.1 in appendix
    def _filtering_stage_basl(self, accepts_x: pd.DataFrame, rejects_x: pd.DataFrame) -> np.ndarray:
        """
        Filters rejected cases that are most and least similar to accepts using Isolation Forest.
        This function mimics the R `filteringStage` and is local to BiasAwareSelfLearning.

        Parameters
        ----------
        accepts_x : pd.DataFrame
            DataFrame containing accepted applications features.
        rejects_x : pd.DataFrame
            DataFrame containing rejected applications features.

        Returns
        -------
        filtered_rejects_x : pd.DataFrame
            DataFrame containing rejected samples that passed the filter.
        """

        # Fit Isolation Forest on accepted data (treating accepts as 'normal' data)
        model = IsolationForest(n_estimators=100, random_state=7)
        model.fit(accepts_x)

        # Get anomaly scores for rejects.
        scores = -model.decision_function(rejects_x)

        # Filter based on percentiles using self.filtering_beta
        lower_bound = np.percentile(scores, self.filtering_beta[0] * 100)
        upper_bound = np.percentile(scores, self.filtering_beta[1] * 100)

        tmp_rejects_mask = (scores >= lower_bound) & (scores <= upper_bound)

        return rejects_x[tmp_rejects_mask].copy()


    #helper function, reference to Algorithm B.2 in appendix
    def _labeling(self, current_train_X: pd.DataFrame, current_train_y: pd.Series, current_unlabeled_X: pd.DataFrame, iteration_number: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, int]:
        """
        This method samples a subset of the currently unlabeled rejected data, trains a weak
        learner on the combined accepted and previously pseudo-labeled data, and then
        assigns confident pseudo-labels (0 or 1) to the sampled unlabeled instances
        based on defined confidence thresholds (`labeling_percent` and `multiplier`).

        Parameters
        ----------
        current_train_X : pd.DataFrame
            The features of the currently labeled training data (accepted samples +
            previously pseudo-labeled rejected samples).
        current_train_y : pd.Series
            The labels corresponding to `current_train_X`.
        current_unlabeled_X : pd.DataFrame
            The features of the rejected samples that are currently unlabeled and
            available for pseudo-labeling in the current iteration.
        iteration_number : int
            The current iteration number of the self-learning loop, used primarily
            for verbose output and debugging.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, int]
            - **`updated_train_X`** (`pd.DataFrame`): The augmented training features,
              including the `current_train_X` and the `newly_labeled_features`.
            - **`updated_train_y`** (`pd.Series`): The augmented training labels,
              including the `current_train_y` and the `newly_labeled_y`.
            - **`updated_unlabeled_X`** (`pd.DataFrame`): The remaining unlabeled
              rejected samples after the current iteration's pseudo-labeling.
              These are the samples from `current_unlabeled_X` that were *not* sampled or *not* confidently pseudo-labeled.
            - **`num_pseudo_labeled`** (`int`): The count of samples that were
              confidently pseudo-labeled in this iteration.
        """

        if current_unlabeled_X.empty:
            if not self.silent:
                print(f"Iteration {iteration_number}: No rejected samples remaining for labeling.")
            return current_train_X, current_train_y, current_unlabeled_X, 0

        # Sample a subset of unlabeled data
        n_unlabeled = len(current_unlabeled_X)
        n_sample = max(1, round(self.sampling_percent * n_unlabeled))

        sampled_indices = random.sample(list(current_unlabeled_X.index), n_sample)
        X_sampled = current_unlabeled_X.loc[sampled_indices].copy()

        # Train weak model and predict probabilities
        current_weak_model = copy.deepcopy(self.weak_learner_estimator)
        current_weak_model.fit(current_train_X, current_train_y)
        s_star_proba = current_weak_model.predict_proba(X_sampled)[:, 1]
        s_star_series = pd.Series(s_star_proba, index=X_sampled.index)

        # Determine confidence thresholds
        # con_g: Threshold for confident GOOD (low prob of being BAD)
        con_g = np.quantile(s_star_series, self.labeling_percent / self.multiplier)

        # con_b: Threshold for confident BAD (high prob of being BAD)
        # This is equivalent to (self.labeling_percent * self.multiplier) percentile from the TOP
        con_b = np.quantile(s_star_series, 1 - self.labeling_percent)

        # Identify confident good (0) and bad (1) samples
        confident_good_mask = (s_star_series <= con_g)
        confident_bad_mask = (s_star_series >= con_b)

        # Ensure masks do not overlap to prevent overwriting. Prioritize 'good' if needed, or exclude overlap.
        # For this implementation, we will only label samples that are exclusively in one confident band.
        exclusively_good_mask = confident_good_mask & ~confident_bad_mask
        exclusively_bad_mask = confident_bad_mask & ~confident_good_mask

        # Combine masks for newly labeled features
        confident_total_mask = exclusively_good_mask | exclusively_bad_mask
        newly_labeled_features = X_sampled[confident_total_mask].copy()

        # Assign pseudo-labels
        newly_labeled_y = pd.Series(index=newly_labeled_features.index, dtype=int)
        newly_labeled_y.loc[s_star_series[exclusively_good_mask].index] = 0
        newly_labeled_y.loc[s_star_series[exclusively_bad_mask].index] = 1

        num_pseudo_labeled = len(newly_labeled_y)

        if num_pseudo_labeled == 0:
            if not self.silent:
                print(f"Iteration {iteration_number}: No confident pseudo-labels found.")
            return current_train_X, current_train_y, current_unlabeled_X, 0

        # Augment training data with newly labeled samples
        updated_train_X = pd.concat([current_train_X, newly_labeled_features], axis=0)
        updated_train_y = pd.concat([current_train_y, newly_labeled_y], axis=0)

        # Update master inferred labels
        self.inferred_labels_.loc[newly_labeled_y.index] = newly_labeled_y.values

        # Remove labeled data from unlabeled pool
        updated_unlabeled_X = current_unlabeled_X.drop(index=newly_labeled_features.index, errors='ignore').copy()

        if not self.silent:
            num_bad = sum(newly_labeled_y == 1)
            num_good = sum(newly_labeled_y == 0)
            print(f"Iteration {iteration_number}: Pseudo-labeled {num_pseudo_labeled} cases ({num_bad} BAD + {num_good} GOOD).")
            print(f"Remaining unlabeled samples: {len(updated_unlabeled_X)}")

        return updated_train_X, updated_train_y, updated_unlabeled_X, num_pseudo_labeled


    # With reference to Algorithm B.3 in Appendix
    def fit(self, accepts_x: pd.DataFrame, accepts_y: pd.Series, rejects_x: pd.DataFrame):
        """
        Builds the Bias-Aware Self-Learning model based on the provided pseudo-code blueprint.

        Parameters
        ----------
        accepts_x : pd.DataFrame
            Features of the accepted (observed) samples.
        accepts_y : pd.Series
            True labels (0/1) of the accepted samples.
        rejects_x : pd.DataFrame
            Features of the rejected samples whose outcomes are unknown.
            This is the *original full set* of rejected data.

        Returns
        -------
        self : object
            Returns the instance itself after fitting.
        """

        # --- Input Validation  ---
        if not isinstance(accepts_x, pd.DataFrame):
            raise TypeError("`accepts_x` must be a pandas DataFrame.")
        if not isinstance(accepts_y, pd.Series):
            raise TypeError("`accepts_y` must be a pandas Series.")
        if accepts_x.empty or accepts_y.empty:
            raise ValueError("`accepts_x` and `accepts_y` cannot be empty. They form the initial labeled dataset.")
        if len(accepts_x) != len(accepts_y):
            raise ValueError("`accepts_x` and `accepts_y` must have the same number of rows.")

        if not isinstance(rejects_x, pd.DataFrame):
            raise TypeError("`rejects_x` must be a pandas DataFrame.")
        if rejects_x.empty:
            raise ValueError("`rejects_x` cannot be empty.")


        # Check column consistency across feature dataframes
        feature_cols = accepts_x.columns.tolist()
        if not rejects_x.empty and feature_cols != rejects_x.columns.tolist():
            raise ValueError("Columns of `rejects_x` must match `accepts_x`.")

        # Set random seeds for reproducibility
        random.seed(7)
        np.random.seed(7)

        # Initialize current training data and unlabeled rejects

        current_train_X, holdout_x_accept, current_train_y, holdout_y_accept = train_test_split(
            accepts_x, accepts_y, test_size=self.holdout_percent,stratify=accepts_y)

        current_unlabeled_X, holdout_x_reject = train_test_split(
            rejects_x, test_size=self.holdout_percent)

        self.inferred_labels_ = pd.Series(index=rejects_x.index, dtype=int).fillna(-1)

        # Apply filtering stage
        current_unlabeled_X = self._filtering_stage_basl(accepts_x=current_train_X, rejects_x=current_unlabeled_X)
        if not self.silent: print(f"Filtering rejects: Kept {len(current_unlabeled_X)} samples.")

        # Initialize iteration tracking and model storage
        iter_count = 0
        perf_vector = []
        fitted_models_at_iterations = []

        # Iterative self-learning loop
        while True:
            iter_count += 1 # Increment iteration count

            # Check stopping criteria (jmax, Xr != empty, Vj >= Vj-1)
            if iter_count > self.max_iterations: # Max iterations reached
                break
            if current_unlabeled_X.empty: # No more unlabeled data
                if not self.silent: print("No unlabeled samples remaining, ending iterations.")
                break

            # Early stopping check based on performance (Vj >= Vj-1)
            # This applies only if early_stop is true AND we have at least two performance points to compare.
            if self.early_stop and len(perf_vector) >= 1 and perf_vector[-1] < perf_vector[-2] if len(perf_vector) >= 2 else False:
                if not self.silent: print(f"--- Iteration {iter_count}: Early stopping (performance dropped). ---")
                break

            if not self.silent: print(f"\n--- Iteration {iter_count} ---")

            # Labeling stage: pseudo-label rejects, update training data (pseudo-code 5-7)
            current_train_X, current_train_y, current_unlabeled_X, num_pseudo_labeled = \
                self._labeling(current_train_X, current_train_y, current_unlabeled_X, iteration_number=iter_count)

            # Training stage: train strong model on augmented data
            current_strong_model_iter = copy.deepcopy(self.strong_estimator)
            current_strong_model_iter.fit(current_train_X, current_train_y)
            fitted_models_at_iterations.append(current_strong_model_iter)

            # Evaluation stage: assess current model performance
            current_perf = -np.inf
            if self.early_stop and not holdout_x_accept.empty and not holdout_x_reject.empty:
                y_pred_acc_holdout = current_strong_model_iter.predict(holdout_x_accept)
                y_proba_acc_holdout = current_strong_model_iter.predict_proba(holdout_x_accept)[:, 1]
                y_pred_rej_holdout = current_strong_model_iter.predict(holdout_x_reject)
                y_proba_rej_holdout = current_strong_model_iter.predict_proba(holdout_x_reject)[:, 1]

                current_perf = self.bayesian_evaluator.BM(
                    y_true_acc=holdout_y_accept.values,
                    y_proba_acc=y_proba_acc_holdout,
                    y_proba_rej=y_proba_rej_holdout,
                    rejects_prior=y_proba_rej_holdout,
                    acc_rate=self.bayesian_acc_rate,
                    seed=7
                )

            perf_vector.append(current_perf)

            if not self.silent and self.early_stop:
                print(f"Iteration {iter_count} strong learner performance: {perf_vector[-1]:.4f}")
                # Log if performance decreased for debugging, actual break handled by while condition
                if len(perf_vector) >=2 and perf_vector[-1] < perf_vector[-2]:
                    print('-- Performance decreased.')

        # Final model selection based on performance
        if self.early_stop and len(perf_vector) > 0:
            best_iter_index = np.argmax(perf_vector)
            self.model_ = fitted_models_at_iterations[best_iter_index]
            if not self.silent:
                # Adjust index for 1-based iteration printing
                print(f"Final model selected from iteration {best_iter_index + 1} (BM: {perf_vector[best_iter_index]:.4f}).")
        else:
            # If no iterations ran, or early stopping wasn't active, use the last fitted model.
            self.model_ = fitted_models_at_iterations[-1] if fitted_models_at_iterations else copy.deepcopy(self.strong_estimator)
            if not hasattr(self.model_, 'n_features_in_'): # Check if fitted
                self.model_.fit(current_train_X, current_train_y)
            if not self.silent and not self.early_stop:
                print("Final model from last iteration (early stopping off).")

        # Post-process inferred labels
        self.inferred_labels_ = self.inferred_labels_.replace({-1: np.nan})

        if not self.silent: print('-- Finished fitting BiasAwareSelfLearning.')
        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X using the fitted BiasAwareSelfLearning model.

        This method uses the 'strong' model which is the final or best-performing model trained during the self-learning process, to make predictions
        on new, unseen data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            The input samples (features) for which to predict class labels.

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the `fit` method has not been called (i.e., the model is not trained yet).
        """
        # Check if the model has been fitted (i.e., self.model_ exists).
        if not hasattr(self, 'model_'):
            raise NotFfittedError("This estimator instance is not fitted yet. "
                                 "Call 'fit' with appropriate arguments before using this method.")
        # Delegate the prediction to the internal fitted strong learner.
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X using the fitted model.

        This method delegates the `predict_proba` call to the internally
        stored strong learner (`self.model_`) which was selected as the
        final model during the `fit` process.

        Parameters
        ----------
        X : np.ndarray
            The input samples for which to predict class probabilities.
            This should be a numpy array or a pandas DataFrame compatible
            with the strong estimator's input requirements.

        Returns
        -------
        np.ndarray
            The predicted class probabilities for each sample.
            The shape of the array will be (n_samples, n_classes),
            where n_classes is typically 2 for binary classification
            (probability of class 0, probability of class 1).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the `fit` method has not been called on this estimator instance
            before `predict_proba` is called.
        AttributeError
            If the strong estimator assigned to `self.model_` does not have
            a `predict_proba` method, which is a requirement for this class.
            eg if it is a regression model or clustering model.
        """

        if not hasattr(self, 'model_'):
            raise NotFittedError("This estimator instance is not fitted yet. "
                                 "Call 'fit' with appropriate arguments before using this method.")
        if not hasattr(self.model_, 'predict_proba'):
            raise AttributeError(f"The strong estimator '{self.strong_estimator.__class__.__name__}' "
                                 "does not support `predict_proba` method.")
        return self.model_.predict_proba(X)

class LabelAllRejectsAsBad(RejectInference):
    """
    Reject Inference Benchmark.

    This benchmark assumes that all rejected applicants are BAD (default = 1).
    It then trains a strong estimator on the combined dataset of:

    - Accepted applicants with their true observed labels (0 = GOOD, 1 = BAD).
    - Rejected applicants artificially labeled as BAD (1).

    The approach is deliberately extreme and biased but provides a
    widely used baseline for comparison against more sophisticated
    reject inference strategies.

    Notes
    -----
    - Compatible with both pandas DataFrames and NumPy arrays.
    - Supports user-provided scikit-learn compatible estimators.
    - If no estimator is provided, defaults to XGBoost or Logistic Regression.
    - Performs input validation (dimension checks, label validation, empties).
    - Uses `sklearn.base.clone` to ensure fresh estimator instances for training.

    """

    def __init__(
        self,
        strong_estimator: Optional[object] = None,
        *,
        estimator: str = "xgb",
        estimator_params: Optional[Dict] = None,
        random_state: int = 1,
        silent: bool = False,
    ):
        """
        Initialize the LARAB model.

        Parameters
        ----------
        strong_estimator : scikit-learn compatible estimator, optional
            A classifier implementing `fit`, `predict`, and `predict_proba`.
            If provided, overrides the default estimator.

        estimator : str, default="xgb"
            Choice of default estimator if `strong_estimator` is None:
            - "xgb"    : XGBClassifier (gradient boosted trees).
            - "logreg" : LogisticRegression.

        estimator_params : dict, optional
            Additional keyword arguments to override the default estimator’s
            hyperparameters.

        random_state : int, default=1
            Random seed for reproducibility.

        silent : bool, default=False
            If True, suppresses all console output during fitting.
        """
        super().__init__()
        self.silent = silent
        self.random_state = random_state
        self.estimator_name = estimator.lower()
        self.estimator_params = estimator_params or {}
        self.strong_estimator = strong_estimator or self._build_default_estimator()
        self.model_: Optional[object] = None

        # Basic checks
        if not hasattr(self.strong_estimator, "fit"):
            raise AttributeError("`strong_estimator` must implement `fit`.")
        if not hasattr(self.strong_estimator, "predict_proba"):
            raise AttributeError("`strong_estimator` must implement `predict_proba`.")
        if not isinstance(self.silent, bool):
            raise ValueError("`silent` must be a boolean.")

    def _build_default_estimator(self):
        """
        Build the default estimator when no external `strong_estimator` is provided.

        Returns
        -------
        scikit-learn compatible estimator
            Instantiated default classifier.
        """
        if self.estimator_name == "xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError as e:
                raise ImportError(
                    "XGBoost is not installed. Install it with `pip install xgboost` "
                    "or provide a different `strong_estimator`."
                ) from e

            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
            }
            params.update(self.estimator_params)
            return XGBClassifier(**params)

        elif self.estimator_name in {"logreg", "logistic", "logisticregression"}:
            from sklearn.linear_model import LogisticRegression
            params = {"max_iter": 1000, "random_state": self.random_state}
            params.update(self.estimator_params)
            return LogisticRegression(**params)

        else:
            raise ValueError(
                f"Unsupported estimator '{self.estimator_name}'. "
                "Use 'xgb' or 'logreg'."
            )

    # ---------- helpers to handle pandas OR numpy ----------
    @staticmethod
    def _to_numpy(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to NumPy array if not already."""
        return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    @staticmethod
    def _to_numpy_1d(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Convert labels to a flattened NumPy array (1D)."""
        y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
        return y_arr.reshape(-1)
    # -------------------------------------------------------

    def fit(self, accepts_x, accepts_y, rejects_x):
        """
        Fit the LARAB model.

        Parameters
        ----------
        accepts_x : pd.DataFrame or np.ndarray
            Features of accepted (labeled) applications.
        accepts_y : pd.Series or np.ndarray
            Labels of accepted applications (0 = GOOD, 1 = BAD).
        rejects_x : pd.DataFrame or np.ndarray
            Features of rejected (unlabeled) applications.

        Returns
        -------
        self : LabelAllRejectsAsBad
            The fitted instance.

        Raises
        ------
        ValueError
            If inputs are empty, misaligned, or labels invalid.
        """
        ax = self._to_numpy(accepts_x)
        ay = self._to_numpy_1d(accepts_y)
        rx = self._to_numpy(rejects_x)

        if ax.shape[0] != ay.shape[0]:
            raise ValueError("`accepts_x` and `accepts_y` must have the same number of rows.")
        if rx.shape[1] != ax.shape[1]:
            raise ValueError("`rejects_x` must have the same number of columns as `accepts_x`.")
        if ax.size == 0 or ay.size == 0:
            raise ValueError("`accepts_x` and `accepts_y` cannot be empty.")
        if rx.size == 0:
            raise ValueError("`rejects_x` cannot be empty.")

        # Validate labels are binary
        unique_labels = np.unique(ay)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("`accepts_y` must only contain binary labels {0,1}.")

        # Label all rejects as BAD (1)
        y_rejects = np.ones(rx.shape[0], dtype=int)

        # Combine accepts and rejects
        X_train = np.vstack([ax, rx])
        y_train = np.concatenate([ay, y_rejects])

        if not self.silent:
            print(f"Training on {X_train.shape[0]} samples: "
                  f"{ax.shape[0]} accepts + {rx.shape[0]} rejects (all labeled BAD).")

        model = clone(self.strong_estimator)
        model.fit(X_train, y_train)
        self.model_ = model

        if not self.silent:
            print("LabelAllRejectsAsBad model training complete.")
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels using the fitted model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix for prediction.

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels (0 = GOOD, 1 = BAD).

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        if self.model_ is None:
            raise NotFittedError("This LabelAllRejectsAsBad instance is not fitted yet. Call `fit` first.")
        return self.model_.predict(self._to_numpy(X))

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities using the fitted model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix for prediction.

        Returns
        -------
        y_proba : np.ndarray
            Predicted probabilities for each class.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        if self.model_ is None:
            raise NotFittedError("This LabelAllRejectsAsBad instance is not fitted yet. Call `fit` first.")
        return self.model_.predict_proba(self._to_numpy(X))

class Reweighting(RejectInference):
    """
    Reject Inference Benchmark.

    This benchmark corrects for selection bias by reweighting accepted applicants
    according to the inverse of their estimated acceptance probability.

    Steps:
    -------
    1. Train an *acceptance model* to predict the probability of being accepted
       (accepts vs rejects).
    2. For each accepted applicant, compute the acceptance probability
       p_hat(a=1|x).
    3. Assign weight = 1 / p_hat for each accepted applicant.
       - Optionally, banded reweighting groups applicants by quantiles of p_hat
         and assigns a constant weight within each band (stabilizes training).
    4. Train the strong estimator (credit scoring model) on the accepted
       applicants using these weights.

    Notes
    -----
    - Only accepted applicants are used for training the scoring model.
    - Rejected applicants do *not* receive labels.
    - This method mitigates but does not completely eliminate selection bias.
    - Supports both pandas DataFrames and NumPy arrays.

    """

    def __init__(
        self,
        strong_estimator: Optional[BaseEstimator] = None,
        acceptance_model: Optional[BaseEstimator] = None,
        *,
        estimator: str = "xgb",
        estimator_params: Optional[Dict] = None,
        banded: bool = False,
        n_bands: int = 10,
        clip_min: float = 1e-3,
        random_state: int = 1,
        silent: bool = False,
    ):
        """
        Initialize the Reweighting model.

        Parameters
        ----------
        strong_estimator : BaseEstimator, optional
            The scoring model (classifier). Must support `fit`, `predict`,
            and `predict_proba`.

        acceptance_model : BaseEstimator, optional
            The model used to estimate acceptance probability.
            Defaults to LogisticRegression if None.

        estimator : str, default="xgb"
            Default scoring estimator if `strong_estimator` is None.
            Choices: 'xgb', 'logreg'.

        estimator_params : dict, optional
            Additional kwargs for the scoring estimator.

        banded : bool, default=False
            If True, use banded reweighting (quantile grouping).

        n_bands : int, default=10
            Number of quantile bands if `banded=True`.

        clip_min : float, default=1e-3
            Minimum acceptance probability before inversion
            (prevents exploding weights).

        random_state : int, default=1
            Random seed.

        silent : bool, default=False
            If True, suppress logging.
        """
        super().__init__()
        self.silent = silent
        self.random_state = random_state
        self.banded = banded
        self.n_bands = n_bands
        self.clip_min = clip_min
        self.estimator_name = estimator.lower()
        self.estimator_params = estimator_params or {}
        self.strong_estimator = strong_estimator or self._build_default_estimator()
        self.acceptance_model = acceptance_model or LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.model_: Optional[BaseEstimator] = None

        # Checks
        if not hasattr(self.strong_estimator, "fit"):
            raise AttributeError("`strong_estimator` must implement `fit`.")
        if not hasattr(self.strong_estimator, "predict_proba"):
            raise AttributeError("`strong_estimator` must implement `predict_proba`.")
        if not hasattr(self.acceptance_model, "predict_proba"):
            raise AttributeError("`acceptance_model` must implement `predict_proba`.")

    def _build_default_estimator(self) -> BaseEstimator:
        """Return default scoring estimator."""
        if self.estimator_name == "xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError as e:
                raise ImportError("XGBoost not installed. Install with `pip install xgboost`.") from e
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
            }
            params.update(self.estimator_params)
            return XGBClassifier(**params)
        elif self.estimator_name in {"logreg", "logistic", "logisticregression"}:
            params = {"max_iter": 1000, "random_state": self.random_state}
            params.update(self.estimator_params)
            return LogisticRegression(**params)
        else:
            raise ValueError("Unsupported estimator. Use 'xgb' or 'logreg'.")

    # -------- helpers for numpy/pandas ----------
    @staticmethod
    def _to_numpy(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    @staticmethod
    def _to_numpy_1d(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
        return y_arr.reshape(-1)
    # --------------------------------------------

    def fit(self, accepts_x, accepts_y, rejects_x):
        """
        Fit the Reweighting model.

        Parameters
        ----------
        accepts_x : pd.DataFrame or np.ndarray
            Features of accepted (labeled) applications.

        accepts_y : pd.Series or np.ndarray
            Labels of accepted applications (0 = GOOD, 1 = BAD).

        rejects_x : pd.DataFrame or np.ndarray
            Features of rejected (unlabeled) applications.

        Returns
        -------
        self : Reweighting
            The fitted instance.
        """
        ax = self._to_numpy(accepts_x)
        ay = self._to_numpy_1d(accepts_y)
        rx = self._to_numpy(rejects_x)

        if ax.shape[0] != ay.shape[0]:
            raise ValueError("`accepts_x` and `accepts_y` must have same rows.")
        if rx.shape[1] != ax.shape[1]:
            raise ValueError("`rejects_x` must have same columns as `accepts_x`.")
        if ax.size == 0 or ay.size == 0:
            raise ValueError("Accepted data cannot be empty.")
        if rx.size == 0:
            raise ValueError("Rejected data cannot be empty.")

        # Validate binary labels
        if not np.all(np.isin(np.unique(ay), [0, 1])):
            raise ValueError("`accepts_y` must contain only binary labels {0,1}.")

        # Build acceptance dataset (accepts=1, rejects=0)
        accept_labels = np.ones(ax.shape[0], dtype=int)
        reject_labels = np.zeros(rx.shape[0], dtype=int)
        X_acceptance = np.vstack([ax, rx])
        y_acceptance = np.concatenate([accept_labels, reject_labels])

        # Train acceptance model
        self.acceptance_model.fit(X_acceptance, y_acceptance)

        # Predict acceptance probs for accepts only
        probs = self.acceptance_model.predict_proba(ax)[:, 1]
        probs = np.clip(probs, self.clip_min, 1.0)

        if self.banded:
            # Assign weights by quantile bands
            quantiles = np.linspace(0, 1, self.n_bands + 1)
            bins = np.digitize(probs, np.quantile(probs, quantiles), right=True)
            weights = np.array([1.0 / np.mean(probs[bins == b]) for b in bins])
        else:
            weights = 1.0 / probs

        # Normalize weights to mean = 1
        weights /= np.mean(weights)

        if not self.silent:
            print(f"Training strong estimator on {ax.shape[0]} accepted applicants "
                  f"with reweighting (banded={self.banded}).")

        model = clone(self.strong_estimator)
        model.fit(ax, ay, sample_weight=weights)
        self.model_ = model

        if not self.silent:
            print("Reweighting model training complete.")
        return self

    def predict(self, X) -> np.ndarray:
        if self.model_ is None:
            raise NotFittedError("Reweighting model not fitted. Call `fit` first.")
        return self.model_.predict(self._to_numpy(X))

    def predict_proba(self, X) -> np.ndarray:
        if self.model_ is None:
            raise NotFittedError("Reweighting model not fitted. Call `fit` first.")
        return self.model_.predict_proba(self._to_numpy(X))


class HeckmanTwoStage(RejectInference):
    """
    Implements the two-stage Heckman correction model for handling sample
    selection bias in binary classification problems.

    The model works in two stages:
    1. A selection model (Probit, Logistic Regression, or XGBoost) is fit on the
       entire population (accepted and rejected applications) to predict the
       probability of acceptance.
    2. An outcome model (Probit or XGBoost) is fit on only the accepted
       population. This model includes an additional variable, the Inverse Mills Ratio (IMR),
       which corrects for the sample selection bias. 
       
    Args:
    selection_classifier (str): The classifier to use for the first stage
        (selection model). Supported options: 'Probit', 'LogisticRegression', 'XGB'.
        Defaults to 'Probit'.
    outcome_classifier (str): The classifier to use for the second stage
        (outcome model). Supported options: 'Probit', 'XGB'.
        Defaults to 'XGB'.
    selection_features_idx (Optional[List[int]]): A list of integer indices
        of the features to be used in the selection model. If None, all
        features are used. Defaults to None.
    outcome_features_idx (Optional[List[int]]): A list of integer indices
        of the features to be used in the outcome model. If None, all
        features are used. Defaults to None.
    """
    
    def __init__(self,
                 selection_classifier: str = 'Probit',
                 outcome_classifier: str = 'XGB',
                 selection_features_idx: Optional[List[int]] = None,
                 outcome_features_idx: Optional[List[int]] = None):

        super().__init__()
        self.selection_classifier = selection_classifier
        self.outcome_classifier = outcome_classifier
        self.selection_model = None
        self.outcome_model = None
        self.selection_features_idx = selection_features_idx
        self.outcome_features_idx = outcome_features_idx


    def fit(self, accepts_x: pd.DataFrame, accepts_y: pd.Series, rejects_x: pd.DataFrame):
        """
        Builds the two-stage Heckman correction model.
        """
        self.feature_columns = accepts_x.columns.tolist()

        # If indices are not provided, use all features
        if self.selection_features_idx is None:
            self.selection_features_idx = list(range(len(self.feature_columns)))
        if self.outcome_features_idx is None:
            self.outcome_features_idx = list(range(len(self.feature_columns)))

        # Check for valid indices
        max_idx = len(self.feature_columns) - 1
        if not all(0 <= idx <= max_idx for idx in self.selection_features_idx + self.outcome_features_idx):
            raise IndexError(f"Feature indices are out of bounds. Valid indices are 0 to {max_idx}.")

        # Stage 1: Selection Equation (Prob(Acceptance))
        print("Stage 1: Fitting selection model on entire population...")
        X_all = pd.concat([accepts_x, rejects_x], ignore_index=True)
        y_selection = pd.Series([1] * len(accepts_x) + [0] * len(rejects_x))

        # Use .iloc to select features by their integer index
        X_selection = X_all.iloc[:, self.selection_features_idx]

        if self.selection_classifier == 'Probit':
            X_selection_sm = sm.add_constant(X_selection, prepend=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.selection_model = sm.Probit(y_selection, X_selection_sm).fit(disp=False, maxiter=5000)
        elif self.selection_classifier == 'LogisticRegression':
            self.selection_model = LogisticRegression(solver='liblinear', penalty='l2', max_iter=1000, random_state=42)
            self.selection_model.fit(X_selection.values, y_selection)
        elif self.selection_classifier == 'XGB':
            self.selection_model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss'
            )
            self.selection_model.fit(X_selection.values, y_selection)
        else:
            raise ValueError(f"Unknown selection_classifier: {self.selection_classifier}")

        # Stage 2: Outcome Equation (Prob(Bad) | Accepted)
        print("Stage 2: Fitting outcome model on accepted population...")
        
        if self.selection_classifier == 'Probit':
            selection_preds = self.selection_model.predict(sm.add_constant(accepts_x.iloc[:, self.selection_features_idx], prepend=False))
        elif self.selection_classifier == 'LogisticRegression':
            selection_preds = self.selection_model.predict_proba(accepts_x.iloc[:, self.selection_features_idx].values)[:, 1]
        elif self.selection_classifier == 'XGB':
            selection_preds = self.selection_model.predict_proba(accepts_x.iloc[:, self.selection_features_idx].values)[:, 1]
        
        #Clip the selection predictions with a larger value to prevent numerical instability issues
        selection_preds_clipped = np.clip(selection_preds, 1e-6, 1 - 1e-6)
        z_values = norm.ppf(selection_preds_clipped)
        imr_values = norm.pdf(z_values) / norm.cdf(z_values)

        # Combine outcome features with IMR
        X_outcome = accepts_x.iloc[:, self.outcome_features_idx]
        X_outcome_with_imr = pd.concat([X_outcome, pd.Series(imr_values, index=X_outcome.index, name='IMR')], axis=1)

        if self.outcome_classifier == 'Probit':
            X_outcome_sm = sm.add_constant(X_outcome_with_imr, prepend=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.outcome_model = sm.Probit(accepts_y, X_outcome_sm).fit(disp=False, maxiter=5000)
        elif self.outcome_classifier == 'XGB':
            self.outcome_model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss'
            )
            self.outcome_model.fit(X_outcome_with_imr.values, accepts_y)
        else:
            raise ValueError(f"Unknown outcome_classifier: {self.outcome_classifier}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the probability of the outcome for new samples.
        """
        if self.selection_model is None or self.outcome_model is None:
            raise NotFittedError("This model instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        # Ensure the input X is a DataFrame to avoid iloc errors
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)

        # Use .iloc to select features by their integer index
        X_selection_df = X.iloc[:, self.selection_features_idx]
        X_outcome_df = X.iloc[:, self.outcome_features_idx]

        # Stage 1: Predict selection probabilities
        if self.selection_classifier == 'Probit':
            selection_preds = self.selection_model.predict(sm.add_constant(X_selection_df, prepend=False))
        elif self.selection_classifier == 'LogisticRegression':
            selection_preds = self.selection_model.predict_proba(X_selection_df.values)[:, 1]
        elif self.selection_classifier == 'XGB':
            selection_preds = self.selection_model.predict_proba(X_selection_df.values)[:, 1]

        # Clip the selection predictions with a larger value to prevent numerical instability issues
        selection_preds_clipped = np.clip(selection_preds, 1e-6, 1 - 1e-6)
        z_values = norm.ppf(selection_preds_clipped)
        imr_values = norm.pdf(z_values) / norm.cdf(z_values)

        # Stage 2: Predict outcome probabilities with IMR
        X_outcome_with_imr = pd.concat([X_outcome_df, pd.Series(imr_values, index=X_outcome_df.index, name='IMR')], axis=1)
        
        if self.outcome_classifier == 'Probit':
            prob_y_eq_1 = self.outcome_model.predict(sm.add_constant(X_outcome_with_imr, prepend=False))
        elif self.outcome_classifier == 'XGB':
            prob_y_eq_1 = self.outcome_model.predict_proba(X_outcome_with_imr.values)[:, 1]
            
        prob_y_eq_0 = 1 - prob_y_eq_1
        
        return np.vstack([prob_y_eq_0, prob_y_eq_1]).T

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for a NumPy array input.
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)
