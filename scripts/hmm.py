import numpy as np
from hmmlearn.base import _BaseHMM
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.naive_bayes import MultinomialNB

def count_transitions(y):
    num_state = np.unique(y).shape[0]
    result = np.zeros((num_state, num_state), dtype=np.uint32)
    for i in range(y.shape[0] - 1):
        result[y[i], y[i + 1]] += 1
    return result


def compute_transition_probs(trans_counts, smoothing=None, alpha=0.01):
    if smoothing == 'add':
        trans_counts = trans_counts + alpha
    div = trans_counts.sum(axis=0)
    return np.divide(trans_counts, div, where=div != 0)


def check_X_ohe(X):
    if X.min() != 0 or X.max() != 1 or X.unique().shape[0] != 2:
        raise Exception('All features in X should be one hor encoded')


def compute_emission_probs_ohe(X, y, smoothing=None, alpha=0.01):
    classes, y = np.unique(y, return_inverse=True)
    Y = y.reshape(-1, 1) == np.arange(len(classes))
    counts = Y.T @ X
    if smoothing == 'add':
        counts = counts + alpha
    div = counts.sum(axis=0)
    return np.divide(counts, div, where=div != 0)


def compute_emission_probs(X, y, smoothing=None, alpha=0.01):
    num_features = np.unique(X).shape[0]
    num_classes = np.unique(y).shape[0]
    result = np.zeros((num_classes, num_features))
    for word, tag in zip(X, y):
        result[tag, word] += 1
    if smoothing == 'add' and alpha is not None:
        result = result + alpha
    div = result.sum(axis=0)
    return np.divide(result, div, where=div != 0)


def compute_init_probs(y):
    _, counts = np.unique(y, return_counts=True)
    return counts / y.shape[0]


def is_ohe(X: np.ndarray):
    return (X.sum(axis=1) == np.ones(len(X))).all() and X.min() == 0 and X.max() == 1


def compute_emission_naive(X, y, prior=None):
    if prior is None:
        nb = MultinomialNB()
    else:
        nb = prior
    nb.fit(X, y)
    return np.exp(nb.coef_)


class CustomHMM(_BaseHMM):
    def __init__(self, n_components=1, startprob_prior=1.0, transmat_prior=1.0, algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False, params=""):
        super().__init__(n_components, startprob_prior, transmat_prior, algorithm, random_state, n_iter, tol, verbose,
                         params)

    def fit(self, X, y=None, lengths=None, smoothing=None):
        if self.params != "" or y is None:
            raise Exception("This implementation is for supervised mode.")

        X = check_array(X)

        self._fit_multinomial(X, y, smoothing)

    def _fit_multinomial(self, X, y, smoothing):
        init_probs = compute_init_probs(y)
        trans_counts = count_transitions(y)
        trans_probs = compute_transition_probs(trans_counts, smoothing=smoothing)

        if is_ohe(X):
            emiss_probs = compute_emission_probs_ohe(X, y, smoothing=smoothing)
        elif X.shape[1] == 1:
            emiss_probs = compute_emission_probs(X, y, smoothing=smoothing)
        else:
            emiss_probs = compute_emission_naive(X, y)

        self.n_components = len(init_probs)
        self.startprob_ = init_probs
        self.transmat_ = trans_probs.T
        self.emissionprob_ = emiss_probs

    def _compute_log_likelihood(self, X):
        return np.log(self.emissionprob_)[:, np.concatenate(X)].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]


if __name__ == '__main__':
    X = np.array([[False, False, False, False, True],
                  [False, False, True, False, False],
                  [False, True, False, False, False],
                  [False, False, True, False, False],
                  [True, False, False, False, False],
                  [True, False, False, False, False],
                  [False, True, False, False, False],
                  [False, False, False, False, True],
                  [False, False, False, True, False],
                  [True, False, False, False, False]])
    # X = np.array([4, 2, 1, 2, 0, 0, 1, 4, 3, 0]).reshape(-1, 1)
    y = np.array([2, 1, 0, 1, 1, 2, 2, 2, 0, 1])
    hmm = CustomHMM(random_state=42)
    hmm.fit(X, y, smoothing='add')
    print(hmm.emissionprob_)
    print(hmm.transmat_)
    print(hmm.startprob_)
    print(hmm.predict(np.array([0, 1, 2, 3]).reshape(-1, 1)))
