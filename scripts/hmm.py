import numpy as np
from hmmlearn.base import _BaseHMM
from sklearn.utils import check_array
from sklearn.utils import check_random_state


def count_transitions(y):
    num_state = np.unique(y).shape[0]
    result = np.zeros((num_state, num_state), dtype=np.uint32)
    for i in range(y.shape[0] - 1):
        result[y[i], y[i + 1]] += 1
    return result


def compute_transition_probs(trans_counts, smoothing=None):
    if smoothing is None:
        return trans_counts / trans_counts.sum(axis=1)[:, np.newaxis]


def check_X_ohe(X):
    if X.min() != 0 or X.max() != 1 or X.unique().shape[0] != 2:
        raise Exception('All features in X should be one hor encoded')


def compute_emission_probs_ohe(X, y, smoothing=None):
    check_X_ohe(X)

    classes, y = np.unique(y, return_inverse=True)
    Y = y.reshape(-1, 1) == np.arange(len(classes))
    counts = Y.T @ X
    if smoothing is None:
        div = counts.sum(axis=1)[:, np.newaxis]
        return np.divide(counts, div, where=div != 0)


def compute_emission_probs(X, y, smoothing=None):
    num_features = np.unique(X).shape[0]
    num_classes = np.unique(y).shape[0]
    result = np.zeros((num_features, num_classes))
    for f, l in zip(X, y):
        result[f, l] += 1
    if smoothing is None:
        div = result.sum(axis=1)[:, np.newaxis]
        return np.divide(result, div, where=div != 0)


def compute_init_probs(y):
    _, counts = np.unique(y, return_counts=True)
    return counts / y.shape[0]


class CustomHMM(_BaseHMM):
    def __init__(self, n_components=1, startprob_prior=1.0, transmat_prior=1.0, algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False, params="", init_params=""):
        super().__init__(n_components, startprob_prior, transmat_prior, algorithm, random_state, n_iter, tol, verbose,
                         params, init_params)

    def fit(self, X, y=None, lengths=None):
        if self.init_params != "" or self.params != "" or y is None:
            raise Exception("This implementation is for supervised mode.")

        X = check_array(X)
        self._check()

        if X.shape[1] == 1:
            self._fit_multinomial(X, y)

    def _fit_multinomial(self, X, y):
        init_probs = compute_init_probs(y)
        trans_counts = count_transitions(y)
        trans_probs = compute_transition_probs(trans_counts, smoothing=None)
        emiss_probs = compute_emission_probs_ohe(X, y)

        self.startprob_ = init_probs
        self.transmat_ = trans_probs
        self.emissionprob_ = emiss_probs
        pass

    def _compute_log_likelihood(self, X):
        return np.log(self.emissionprob_)[:, np.concatenate(X)].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]
