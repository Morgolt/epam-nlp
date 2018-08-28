import numpy as np
from seqlearn.evaluation import bio_f_score
from seqlearn.hmm import MultinomialHMM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_validate

from data import *
from epam_nlp import CustomHMM, get_bio_f1

DATA_PATH = Path('../data')
RAW_DATA_PATH = DATA_PATH / 'processed.tsv'
df = load_data(RAW_DATA_PATH, nrows=1000)
X, y, lengths = get_X_y_lengths(df, cols_to_keep={'token'})
le = LabelEncoder()
ohe = OneHotEncoder(handle_unknown='ignore')
clf = CustomHMM(y=y)
pipeline = Pipeline([
    ('one_hot', ohe),
    ('hmm', clf)
])
cv = get_cv(lengths=lengths)

res = cross_validate(pipeline, X.reshape(-1, 1), y, cv=cv, n_jobs=1, scoring=get_bio_f1)
print(res)

# cv = get_cv(X, y, lengths)
# i = 1
# scores = []
# fscores = []
# print('Fold\tAccuracy\tF-score')
# for train_x, train_len, train_y, test_x, test_len, test_y in cv:
#     hmm = MultinomialHMM()
#     hmm.fit(train_x, y=train_y, lengths=train_len)
#     score = hmm.score(test_x, test_y, test_len)
#
#     pred = hmm.predict(test_x, test_len)
#     str_true = np.asarray(pd.Categorical.from_codes(test_y, df[TARGET].cat.categories), dtype=str)
#     str_pred = np.asarray(pd.Categorical.from_codes(pred, df[TARGET].cat.categories), dtype=str)
#
#     fscore = bio_f_score(str_true, str_pred)
#     fscores.append(fscore)
#
#     print(f'{i}\t{score}\t{fscore}')
#     scores.append(score)
#     i += 1
#
# print(f'CV accuracy: {np.mean(scores)}')
#
# print(f'CV bio f-score: {np.mean(fscores)}')
