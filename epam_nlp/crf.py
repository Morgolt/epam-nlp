from pathlib import Path

import pandas as pd
import sklearn_crfsuite
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn_crfsuite import metrics


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


if __name__ == '__main__':
    PATH = Path('../data') / 'basic_processing.csv'
    CRF_FEATURES = PATH / '..' / 'crf_features_bin'
    SEED = 42
    df = pd.read_csv(PATH)
    data = get_features(df)

    pd.DataFrame.from_dict(data[0])
    # if not CRF_FEATURES.exists():
    #     df = pd.read_csv(PATH)
    #     data = get_features(df)
    #     joblib.dump(data, CRF_FEATURES)
    # features, y, labels = joblib.load(CRF_FEATURES)
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100, all_possible_transitions=True)
    # params = {'c1': [0.01], 'c2': [0.01]}
    # # todo: bio-scorer, analysis of output
    # f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)
    # # search
    # rs = GridSearchCV(crf,
    #                   params,
    #                   cv=5,
    #                   verbose=1,
    #                   n_jobs=-1,
    #                   scoring=f1_scorer)
    # rs.fit(features, y)
    # print('best params:', rs.best_params_)
    # print('best CV score:', rs.best_score_)
