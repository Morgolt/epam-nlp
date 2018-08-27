import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


def get_wordcloud(tokens: pd.Series, seed, title=None):
    wordcloud = WordCloud(collocations=False, width=1200, height=800, random_state=seed).generate_from_frequencies(
        tokens.value_counts())

    _, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    if title is not None:
        ax.title(title)
    ax.axis("off")
    return ax
