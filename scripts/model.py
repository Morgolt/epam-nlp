import numpy as np
from seqlearn.evaluation import bio_f_score
from seqlearn.hmm import MultinomialHMM

from data import *

DATA_PATH = Path('../data')
RAW_DATA_PATH = DATA_PATH / 'raw.tsv'
df = load_data(RAW_DATA_PATH, nrows=100000)

cv = get_cv(df, 5, ['position', 'pos', 'ner', 'part', 'document'], one_hot=True)
i = 1
scores = []
for train_x, train_len, train_y, test_x, test_len, test_y in cv:
    hmm = MultinomialHMM()
    hmm.fit(train_x, y=train_y, lengths=train_len)
    score = hmm.score(test_x, test_y, test_len)
    pred = hmm.predict(test_x, test_len)

    print(f'Fold {i} accuracy: {score}')
    scores.append(score)
    i += 1

print(f'CV accuracy: {np.mean(scores)}')
