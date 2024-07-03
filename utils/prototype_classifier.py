import numpy as np
import pandas as pd
import gensim
from gensim import models, similarities
from collections import defaultdict
import preprocessing
from preprocessing import corpus_extractor
import similarity_ordering

corpus = corpus_extractor()

def similarity_order(corpus_tfidf: gensim.interfaces.TransformedCorpus, dictionary: gensim.corpora.dictionary.Dictionary, queries:list[str], mod, num_topics:int):
    """ Take in a tfidf corpus and dictionary created by corpus_creation,
    as well as a query such as 'customer support' and a model. Then we 
    classify each document in the corpus according to which query
    had the highest similarity score
    This remains a naive classifier, more refinement is needed. LSI model is recommended

    mod: must be formatted as models.ModelName, such as models.LdaModel, or models.LsiModel
    """

    model = mod(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    
    query_scores = []
    
    df = pd.DataFrame(columns = queries)
    
    for q in queries:
        vec_bow = dictionary.doc2bow(q.lower().split())
        vec_model = model[vec_bow]  # convert the query to LSI space

        #index these
        index = similarities.MatrixSimilarity(model[corpus_tfidf])

        sims = index[vec_model]  # perform a similarity query against the corpus
        
        query_scores.append(sims)
        
        df[q] = sims

    df["class"] = df.idxmax(axis=1)
    
    df["text"] = corpus
    
    queries = ["Customer service", "Delivery service", "Product Quality"]

    bins = []

    for q in queries:
        subset = df[df["class"]==q]
        bins.append(subset["text"].values)
    
    classes_dict = {}
    for q, b in zip(queries, bins):
        key, value = q, b
        classes_dict[key] = value
        
    
    return classes_dict