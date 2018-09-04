from pathlib import Path

import pandas as pd
import sklearn_crfsuite
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from epam_nlp.eval import bio_f1_crf


def get_features(df: pd.DataFrame, seq=('part', 'document')):
    grouped = df.groupby(seq)
    data = []
    y = []
    for _, group in grouped:
        doc_data = []
        doc_y = []
        for i, row in enumerate(group.itertuples()):
            features = dict(
                bias=1.0,
                pos=row.pos,
                shape=row.shape,
                stem=row.stem
            )
            if i > 0:
                prev = group.iloc[i - 1]
                features.update(dict(
                    prev_lemma=prev['lemma'],
                    prev_shape=prev['shape'],
                    prev_pos=prev['pos']
                ))
            else:
                features['BOS'] = True

            if i < group.shape[0] - 1:
                nex = group.iloc[i + 1]
                features.update(dict(
                    next_lemma=nex['lemma'],
                    next_shape=nex['shape'],
                    next_pos=nex['pos']
                ))
            else:
                features['EOS'] = True
            doc_data.append(features)
            doc_y.append(row.iob_ner)
        data.append(doc_data)
        y.append(doc_y)

    labels = list(df.iob_ner.unique())
    return data, y, labels


def load_crf_features(bin_feat: Path, csv_path=None):
    if not bin_feat.exists():
        df = pd.read_csv(csv_path)
        data = get_features(df)
        joblib.dump(data, bin_feat)
    features, y, labels = joblib.load(bin_feat)
    return features, y, labels


def get_crf(algorithm='lbfgs', max_iterations=100, all_possible_transitions=True, c1=0.01, c2=0.01):
    crf = sklearn_crfsuite.CRF(algorithm=algorithm,
                               max_iterations=max_iterations,
                               all_possible_transitions=all_possible_transitions, c1=c1, c2=c2)
    return crf


def crf_cv(X, y, cv=None, n_jobs=1, verbose=0, **kwargs):
    crf = get_crf()
    f1_scorer = make_scorer(bio_f1_crf)
    return cross_val_score(crf, X, y, scoring=f1_scorer, cv=cv, n_jobs=n_jobs, fit_params=kwargs, verbose=verbose)
