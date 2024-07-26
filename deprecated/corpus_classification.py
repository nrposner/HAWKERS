import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models, similarities
from collections import defaultdict



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

    return corpus_tfidf, dictionary

def similarity_order_2(corpus_tfidf: gensim.interfaces.TransformedCorpus, dictionary: gensim.corpora.dictionary.Dictionary, queries:list[str], mod, num_topics:int):
    """ Take in a tfidf corpus and dictionary created by corpus_creation,
    as well as a query such as 'customer support' and a model. Then we 
    classify each document in the corpus according to which query
    had the highest similarity score
    This remains a naive classifier, more refinement is needed. LSI model is recommended

    mod: must be formatted as models.ModelName, such as models.LdaModel, or models.LsiModel
    """

    model = mod(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    
    
    
    df = pd.DataFrame(columns = queries)
    
    
    #want to output a single number for each corpus element, the mean
    #score across all queries
    
    for q in queries:
        vec_bow = dictionary.doc2bow(q.lower().split())
        vec_model = model[vec_bow]  # convert the query to LSI space

        #index these
        index = similarities.MatrixSimilarity(model[corpus_tfidf])

        sims = index[vec_model]  # perform a similarity query against the corpus
        
        df[q] = sims

        
    scores = df.mean(axis=1)
    
    return scores
    
def classification_pipeline_2(train_corpus, query_sets, mod, num_topics):
    
    #query sets should bea dictionary of lists
    
    
    processed_corpus = preprocess(train_corpus)
    corpus_tfidf, dictionary = corpus_maker(processed_corpus)
    
    df = pd.DataFrame(columns = query_sets.keys()[0]) ##check
    
    
    for item in query_sets.items():
        scores = similarity_order_2(corpus_tfidf, dictionary, item[1], mod, num_topics)
        df[item[0]] = scores
    
    
    df["text"] = train_corpus
    df["class"] = df.idxmax(axis=1)


    bins = []
    
    ###

    for q in query_sets.keys()[0]:
        subset = df[df["class"]==q]
        bins.append(subset["text"].values)
    
    classes_dict = {}
    for q, b in zip(query_sets.keys()[0], bins):
        key, value = q, b
        classes_dict[key] = value
        
    return classes_dict