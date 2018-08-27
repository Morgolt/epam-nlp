import numpy as np
import pandas as pd
from seqlearn.evaluation import bio_f_score
from sklearn_crfsuite.metrics import flat_classification_report


def get_bio_f1(y_true, y_pred, label_mapping):
    str_true, str_pred = int_to_str(y_true, y_pred, label_mapping)
    fscore = bio_f_score(str_true, str_pred)
    return fscore


def int_to_str(y_true, y_pred, label_mapping):
    if isinstance(label_mapping, pd.Series):
        str_true = np.asarray(pd.Categorical.from_codes(y_true, label_mapping.cat.categories), dtype=str)
        str_pred = np.asarray(pd.Categorical.from_codes(y_pred, label_mapping.cat.categories), dtype=str)
    return str_true, str_pred


def get_report(y_true, y_pred, label_mapping):
    str_true, str_pred = int_to_str(y_true, y_pred, label_mapping)
    return flat_classification_report(y_pred=str_pred, y_true=str_true)
