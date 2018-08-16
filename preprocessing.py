import pandas as pd
from pathlib import Path

from pandas.errors import MergeError

DATA_PATH = Path('./data')

# result = pd.DataFrame(columns=['token', 'position', 'pos', 'ner', 'part', 'document'])
res = []
for part in DATA_PATH.iterdir():
    for doc in part.iterdir():
        print(doc)
        tags = pd.read_csv(doc / 'en.tags', sep='\t', header=None, quoting=3,
                           names=['token', 'pos', 'lemma', 'ner', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6'])
        off = pd.read_csv(doc / 'en.tok.off', sep='\s+', header=None, quoting=3,
                          names=['start_character_offset', 'end_character_offset', 'position', 'token'])
        try:
            merged = pd.merge(off, tags, left_index=True, right_index=True, validate='one_to_one').loc[:,
                     ['token_x', 'position', 'pos', 'ner']]
        except MergeError:
            print('merge')
        merged['part'] = part.parts[-1]
        merged['document'] = doc.parts[-1]
        merged.rename(columns={'token_x': 'token'})
        res.append(merged)

df = pd.concat(res)
df.columns = ['token', 'position', 'pos', 'ner', 'part', 'document']
df.to_csv(DATA_PATH / 'raw.tsv', sep='\t', header=True, index=False)
