from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud


def get_wordcloud(tokens: pd.Series, seed, title=None):
    wordcloud = WordCloud(collocations=False, width=1200, height=800, random_state=seed).generate_from_frequencies(
        tokens.value_counts())

    _, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    if title is not None:
        ax.title(title)
    ax.axis("off")
    return ax


def entity_to_idx(values: np.ndarray, default=0):
    dd = defaultdict(lambda: default)
    for v, k in enumerate(np.unique(values)):
        dd[k] = v + 1
    return dd


class UnknownWordsLabelEncoder(LabelEncoder):
    def __init__(self, unknown_list=None, unknown_mapping=None) -> None:
        self.unknown_mapping = unknown_mapping
        self.unknown_list = unknown_list

    def fit_transform(self, X, y=None):
        if X.ndim == 1:
            self.classes_ = np.unique(X)
            self.classes_ = np.sort(np.append(self.classes_, self.unknown_list))
            return self.transform(X).reshape(-1, 1)
        else:
            return self.encode_matrix(X)

    def transform(self, y):
        if y.ndim == 1:
            classes = np.unique(y)
            if len(np.intersect1d(classes, self.classes_)) < len(classes):
                diff = np.setdiff1d(classes, self.classes_)
                unkn_idx = np.isin(y, diff)
                unkn_encoded = self.unknown_mapping.umap(y[unkn_idx])
                y = y.astype(object)
                y[unkn_idx] = unkn_encoded
            return super().transform(y).reshape(-1, 1)
        else:
            return self.transform_matrix(y)

    def inverse_transform(self, y):
        if y.ndim == 2 and y.shape[1] == 1:
            y = np.ravel(y)
            return super().inverse_transform(y)

    def encode_matrix(self, X):
        self.encoders = []
        encoded = []
        for i in range(X.shape[1]):
            col = X[:, i]
            le = UnknownWordsLabelEncoder(self.unknown_list, self.unknown_mapping)
            col_enc = le.fit_transform(col)
            self.encoders.append(le)
            encoded.append(col_enc)
        return np.hstack(encoded)

    def transform_matrix(self, y):
        return np.array([self.encoders[i].transform(col).reshape(-1) for i, col in enumerate(y.T)]).T
