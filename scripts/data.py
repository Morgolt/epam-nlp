from pathlib import Path

import pandas as pd
from seqlearn.evaluation import SequenceKFold

SEQUENCE = ['document', 'part']
TARGET = 'ner'


def load_data(path: Path, nrows=None):
    df = pd.read_csv(path, delimiter='\t', nrows=nrows, header=0)
    df['ner'] = df.ner.str.slice(0, 3)
    df['sentence_number'] = df['position'] // 1000
    # todo: sequence
    df['seq'] = df['']
    df = df.astype('category')
    return df


def get_X_y_lengths(df: pd.DataFrame, cols_to_drop=None, sequence_column=SEQUENCE, one_hot=False):
    if isinstance(sequence_column, str):
        sequence_column = [sequence_column]
    if cols_to_drop is None:
        cols_to_drop = []
    y = df.ner.values.codes.copy()
    lengths = df.groupby(sequence_column, sort=False).count().iloc[:, 0].values
    if TARGET not in cols_to_drop:
        cols_to_drop.append(TARGET)
    X = df.drop(cols_to_drop, axis=1)
    if one_hot:
        X = pd.get_dummies(X).values
    else:
        X = X.values

    return X, y, lengths


def get_cv(df, n_folds=5, cols_to_drop=None, one_hot=False):
    if cols_to_drop is None:
        cols_to_drop = []
    X, y, lengths = get_X_y_lengths(df, cols_to_drop, SEQUENCE, one_hot=one_hot)
    if not set(SEQUENCE).issubset(set(df.columns)):
        lengths = X.shape[0]
    kf = SequenceKFold(lengths=lengths, n_folds=n_folds)
    for (train_ind, train_lengths, test_ind, test_lengths) in kf:
        yield X[train_ind], train_lengths, y[train_ind], X[test_ind], test_lengths, y[test_ind]


if __name__ == '__main__':
    DATA_PATH = Path('../data')
    RAW_DATA_PATH = DATA_PATH / 'raw.tsv'
    df = load_data(RAW_DATA_PATH, nrows=1000)
    X, y, lengths = get_X_y_lengths(df, ['position', 'pos', 'ner', 'part', 'document'])
    cv = get_cv(df, cols_to_drop=['position', 'pos', 'ner', 'part', 'document'])
