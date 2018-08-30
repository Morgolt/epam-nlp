from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
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
        X = column_or_1d(X, warn=True)
        self.classes_ = np.unique(X)
        self.classes_ = np.sort(np.append(self.classes_, self.unknown_list))
        return self.transform(X).reshape(-1, 1)

    def transform(self, y):
        y = column_or_1d(y, warn=True)
        classes = np.unique(y)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            unkn_idx = np.isin(y, diff)
            unkn_encoded = self.unknown_mapping.umap(y[unkn_idx])
            y = y.astype(object)
            y[unkn_idx] = unkn_encoded
        return super().transform(y).reshape(-1, 1)

    def inverse_transform(self, y):
        if y.ndim == 2:
            y = np.ravel(y)
        return super().inverse_transform(y)
