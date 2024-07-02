import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models
from collections import defaultdict

def corpus_maker(processed_corpus:list):
    """ Take in a processed corpus from preprocessing and transform it into
    a tfidf bag of words corpus
    
    """
    # turn this into a dictionary structure
    dictionary = corpora.Dictionary(processed_corpus)

    # create a 'bag of words' corpus using that dictionary
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # train the model

    # tfidf is a transformation that finds term frequency in model frequency
    # we will use this in order to create a structure which other models can attack more easily
    tfidf = models.TfidfModel(bow_corpus)

    corpus_tfidf = tfidf[bow_corpus]

    return corpus_tfidf