from io import StringIO
from pathlib import Path

import pandas as pd


def load_data(path):
    res = []
    for part in path.iterdir():
        for doc in part.iterdir():
            print(doc)
            with (doc / 'en.met').open('r', encoding='utf-8') as fh:
                is_voa = 'Voice of America' in fh.read()
            if is_voa:
                with (doc / 'en.tags').open('r', encoding='utf-8') as fh:
                    content = fh.read().strip()
                    sentences = content.split('\n\n')
                    for ind, sent in enumerate(sentences):
                        tokens = pd.read_csv(StringIO(sent), sep='\t', header=None, quoting=3,
                                             names=['token', 'pos', 'lemma', 'ner', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6'])
                        # parse only first part of NER tag
                        tokens['ner'] = tokens.ner.str.split('-').str[0]
                        tokens = to_conll_iob(tokens)
                        tokens['part'] = part.parts[-1]
                        tokens['document'] = doc.parts[-1]
                        tokens['sentence'] = ind
                        res.append(tokens)
                        del tokens
    df = pd.concat(res)
    return df


def to_conll_iob(df: pd.DataFrame):
    prev_row = tuple()
    result = pd.Series(index=df.index)
    for row in df.itertuples():
        ner = row.ner
        prev_ner = prev_row.ner if len(prev_row) > 0 else None
        ind = row.Index
        if ner != 'O':
            if prev_ner is None:
                result[ind] = "B-" + ner
            elif prev_ner == ner:
                result[ind] = "I-" + ner
            else:
                result[ind] = "B-" + ner
        else:
            result[ind] = ner
        prev_row = row

    df['iob_ner'] = result
    # df.drop('ner', axis=1, inplace=True)
    return df


if __name__ == '__main__':
    RAW_DATA_PATH = Path('../gmb-2.2.0 - Copy/data')
    DATA_PATH = Path('../data')
    df = load_data(RAW_DATA_PATH)
    df.to_csv(DATA_PATH / 'processed_voa.tsv', sep='\t', header=True, index=False,
              columns=['token', 'pos', 'lemma', 'part', 'document', 'sentence', 'ner', 'iob_ner'])
