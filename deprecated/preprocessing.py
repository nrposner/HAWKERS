import numpy as np
import pandas as pd
import gensim
from collections import defaultdict

def corpus_extractor():
    """Pulls out the ninth column from the dataset in order to get
    the raw corpus which will be use din preprocessing
    """
    data = pd.read_csv("data/service_reviews_15000rows_translated.csv")
    corpus = data.iloc[:, 8]
    return corpus

def preprocess(corpus:pd.core.series.Series, min_len:int = 3, max_len:int = 15) -> list:
    """ Take in a corpus of text in a pandas series and perform
    preprocessing

    corpus: a pandas series containing text

    min_len: minimum word length. No shorter words will be retained

    max_len: maximum word length. No longer words will be retained
    """
    
    if not (min_len <= max_len):
        raise ValueError("make sure your minimum and maximum token lengths are not reversed")

    preprocessed_corpus = []

    for i in corpus:
        preprocessed_doc = gensim.utils.simple_preprocess(i, min_len = min_len, max_len = max_len)
    
        preprocessed_corpus.append(preprocessed_doc)

        # go line by line, removing common words
    stoplist = set('for a of the and to in'.split(' '))
    texts = [[word for word in document if word not in stoplist]
         for document in preprocessed_corpus]

    # count word frequencies
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

    return processed_corpus