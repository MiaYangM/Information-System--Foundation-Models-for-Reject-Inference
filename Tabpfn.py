
# Safe TabPFN training helper (paste into notebook)
import numpy as np
import inspect
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

def train_tabpfn(X_train, y_train, X_test,
                 timeout_minutes=30, device="cuda",
                 desired_kwargs=None, seed=None, verbose=1):

    if desired_kwargs is None:
        desired_kwargs = {}

    # Safe dtypes & contiguous arrays
    X_train = np.ascontiguousarray(X_train.astype(np.float32))
    X_test  = np.ascontiguousarray(X_test.astype(np.float32))
    y_train = np.ascontiguousarray(y_train.astype(np.int64))

    # Optionally set global seed for reproducibility (if TabPFN respects numpy seed)
    if seed is not None:
        np.random.seed(int(seed))

    # Create classifier via factory (adjust call if your API differs)
    # Remove duplicate `clf = clf =` typo from your original code
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, ignore_pretraining_limits=True)

    # If your TabPFN supports passing device/verbose/other params, filter desired_kwargs based on signature
    try:
        sig = inspect.signature(TabPFNClassifier.create_default_for_version)
        # Note: factory signature may not accept extra kwargs; skip if not supported.
    except Exception:
        pass

    # Fit. Try to pass timeout if supported by fit
    fit_sig = inspect.signature(clf.fit)
    fit_kwargs = {}
    if 'timeout' in fit_sig.parameters:
        fit_kwargs['timeout'] = int(60 * timeout_minutes)
    elif 'timeout_minutes' in fit_sig.parameters:
        fit_kwargs['timeout_minutes'] = timeout_minutes

    # Optionally print info
    if verbose:
        print(f"Training TabPFN on {X_train.shape[0]} samples, evaluating on {X_test.shape[0]} samples.")
        if fit_kwargs:
            print("Passing fit kwargs:", fit_kwargs)

    try:
        clf.fit(X_train, y_train, **fit_kwargs)
    except TypeError as e:
        # If timeout kw caused the problem, retry without it
        if fit_kwargs:
            if verbose:
                print("fit(...) TypeError with timeout kwargs; retrying without timeout. Error:", e)
            clf.fit(X_train, y_train)
        else:
            # re-raise so caller can see the problem
            raise

    # predict_proba -> ensure we return positive-class 1D array
    probs = clf.predict_proba(X_test)
    if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] > 1:
        probs_pos = probs[:, 1]
    else:
        probs_pos = np.asarray(probs).ravel()

    return probs_pos, clf