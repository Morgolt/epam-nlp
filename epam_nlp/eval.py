import numpy as np
import pandas as pd
from seqlearn.evaluation import bio_f_score
from sklearn_crfsuite.metrics import flat_classification_report


def get_bio_f1(y_true, y_pred, label_mapping=None):
    str_true, str_pred = int_to_str(y_true, y_pred, label_mapping)
    fscore = bio_f_score(str_true, str_pred)
    return fscore


def bio_f1_crf(y_true, y_pred):
    y_true = flatten_y(y_true)
    y_pred = flatten_y(y_pred)
    return bio_f_score(y_true, y_pred)


def flatten_y(y):
    return [item for sublist in y for item in sublist]


def int_to_str(y_true, y_pred, label_mapping=None):
    if isinstance(y_true, list):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
    if isinstance(label_mapping, pd.Series):
        str_true = np.asarray(pd.Categorical.from_codes(y_true, label_mapping.cat.categories), dtype=str)
        str_pred = np.asarray(pd.Categorical.from_codes(y_pred, label_mapping.cat.categories), dtype=str)

    return str_true, str_pred


def get_report(y_true, y_pred, label_mapping):
    str_true, str_pred = int_to_str(y_true, y_pred, label_mapping)
    return flat_classification_report(y_pred=y_pred.reshape(-1, 1), y_true=y_true.reshape(-1, 1),
                                      labels=np.unique(y_true))


def vocabulary_transfer():
    pass
