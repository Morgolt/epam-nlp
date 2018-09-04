import gc

import numpy as np
import sklearn
from hmmlearn.base import _BaseHMM
from scipy.sparse import issparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import check_random_state

from .eval import get_bio_f1
from .util import entity_to_idx


def compute_transition_probs(trans_counts, smoothing=None, alpha=0.01):
    if smoothing == 'add':
        trans_counts = trans_counts + alpha
    div = trans_counts.sum(axis=0)
    return np.divide(trans_counts, div, where=div != 0)


def hmm_cv(X, y, cv, target_mapping, transformer=None, verbose=0):
    scores = []
    i = 1
    num_classes = len(np.unique(y))
    for (train_ind, train_len, test_ind, test_len) in cv:
        hmm = CustomHMM(n_components=num_classes)
        x_train = X[train_ind]
        y_train = y[train_ind]
        if transformer is not None:
            current_transformer = sklearn.base.clone(transformer)
            x_train = current_transformer.fit_transform(x_train)
        hmm.fit(x_train, y=y_train, smoothing='add')

        x_test = X[test_ind]
        if transformer is not None:
            x_test = current_transformer.transform(x_test)
        y_test = y[test_ind]
        if issparse(x_test):
            y_pred = hmm.predict(x_test.toarray())
        else:
            y_pred = hmm.predict(x_test)

        f1 = get_bio_f1(y_test, y_pred, target_mapping)
        if verbose > 0:
            print(f"Fold {i} score: {f1}")
        i += 1
        scores.append(f1)

        del x_train
        del y_train
        del x_test
        del y_test
        del y_pred
        del current_transformer
        gc.collect()
    if verbose > 0:
        print(f"AVG cross-validation f1 score: {np.mean(scores)}")
    return scores


class CustomHMM(_BaseHMM):
    def __init__(self, n_components=1, startprob_prior=1.0, transmat_prior=1.0, algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False, params=""):
        super().__init__(n_components, startprob_prior, transmat_prior, algorithm, random_state, n_iter, tol, verbose,
                         params)

    def fit(self, X, y=None, lengths=None, smoothing='add'):
        if self.params != "" or y is None:
            raise Exception("This implementation is for supervised mode.")

        if X.dtype == object:
            self.word_mapping = entity_to_idx(X)

        self._fit_multinomial(X, y, smoothing)

    def _fit_multinomial(self, X, y, smoothing):
        init_probs = self.compute_init_probs(y)
        trans_counts = self.count_transitions(y)
        trans_probs = compute_transition_probs(trans_counts, smoothing=smoothing)

        if X.ndim == 1:
            emiss_probs = self.compute_emission_probs(X, y, smoothing=smoothing)
        else:
            emiss_probs = self.compute_emission_naive(X, y)

        self.startprob_ = init_probs
        self.transmat_ = trans_probs.T
        self.emissionprob_ = emiss_probs

    def _compute_log_likelihood(self, X):
        pass
        if hasattr(self, 'nb'):
            return self.nb.predict_log_proba(X)
        else:
            return X.dot(np.log(self.emissionprob_.T))

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]

    def compute_emission_probs(self, X, y, smoothing=None, alpha=0.01):
        num_features = len(self.word_mapping.keys()) + 1
        num_classes = len(np.unique(y))
        result = np.zeros((num_classes, num_features))
        for word, tag_idx in zip(X, y):
            word_idx = self.word_mapping[word]
            result[tag_idx, word_idx] += 1
        if smoothing == 'add' and alpha is not None:
            result = result + alpha
        div = result.sum(axis=0)
        return np.divide(result, div, where=div != 0)

    def compute_emission_naive(self, X, y, prior=None):
        if prior is None:
            nb = MultinomialNB()
        else:
            nb = prior
        nb.fit(X, y)
        self.nb = nb
        return np.exp(nb.coef_)

    def predict(self, X, lengths=None):
        if X.ndim == 1:
            X = np.array([self.word_mapping[word] for word in X]).reshape(-1, 1)
        if issparse(X):
            X = X.toarray()
        return super().predict(X, lengths)

    def compute_init_probs(self, y, alpha=0.01):
        cls, counts = np.unique(y, return_counts=True)
        init_probs = np.zeros(self.n_components)
        for c, count in zip(cls, counts):
            init_probs[c] = count
        init_probs = init_probs + alpha
        div = init_probs.sum(axis=0)
        return np.divide(init_probs, div, where=div != 0)

    def count_transitions(self, y):
        result = np.zeros((self.n_components, self.n_components), dtype=np.uint32)
        for i in range(y.shape[0] - 1):
            result[y[i], y[i + 1]] += 1
        return result
