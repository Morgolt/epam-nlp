import re
import string
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from seqlearn.evaluation import SequenceKFold

SEQUENCE = ('document', 'part')
TARGET = 'iob_ner'
PUNCTUATION = pd.Series(list(string.punctuation))


def load_data(path: Path, nrows=None, grouper=SEQUENCE) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter='\t', nrows=nrows)
    df['sentence'] = df['sentence'] + 1
    df['seq'] = df.groupby(list(grouper)).grouper.group_info[0]
    df = df.astype('category')
    return df


def get_utility_token_index(tokens: pd.Series, punctuation=PUNCTUATION):
    return tokens.isin(punctuation) | tokens.isin(stopwords.words('english'))


def get_stem(tokens: pd.Series):
    stemmer = nltk.stem.SnowballStemmer('english')
    return tokens.map(lambda t: stemmer.stem(t))


def get_X_y_lengths(df: pd.DataFrame, cols_to_keep=None, sequence_column='seq', target=TARGET, one_hot=False):
    if isinstance(sequence_column, str):
        sequence_column = [sequence_column]
    if cols_to_keep is None:
        cols_to_keep = {}
    y = df[target].cat.codes.values.copy()
    lengths = df.groupby(sequence_column, sort=False).count().iloc[:, 0].values
    if target in cols_to_keep:
        cols_to_keep.remove(target)
    cols_to_drop = set(df.columns) - cols_to_keep
    X = df.drop(cols_to_drop, axis=1)
    if one_hot:
        X = csr_matrix(pd.get_dummies(X, sparse=True).to_coo())
    else:
        X = X.values if X.shape[1] > 1 else X.iloc[:, 0].values.get_values()
    return X, y, lengths


def get_cv(lengths=None, seed=42, shuffle=True, n_folds=5, yield_lengths=False):
    if lengths is None:
        lengths = np.ones(X.shape[0])
    kf = SequenceKFold(lengths=lengths,
                       n_folds=n_folds,
                       random_state=seed,
                       shuffle=shuffle,
                       yield_lengths=yield_lengths)
    return kf


def word_shape(word):
    t1 = re.sub('[A-Z]', 'X', word)
    t2 = re.sub('[a-z]', 'x', t1)
    return re.sub('[0-9]', 'd', t2)


if __name__ == '__main__':
    DATA_PATH = Path('../data')
    PROCESSED = DATA_PATH / 'processed.tsv'
    df = load_data(PROCESSED, nrows=1000)
    X, y, lengths = get_X_y_lengths(df, {'token'})
    cv = get_cv(lengths=lengths)
