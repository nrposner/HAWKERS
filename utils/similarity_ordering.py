import gensim
from gensim import models, similarities
from collections import defaultdict


def similarity_order(corpus_tfidf: gensim.interfaces.TransformedCorpus, dictionary: gensim.corpora.dictionary.Dictionary, query:str, mod, num_topics:int):
    """ Take in a tfidf corpus and dictionary created by corpus_creation,
    as well as a query such as 'customer support' and a model. 

    mod: must be formatted as models.ModelName, such as models.LdaModel, or models.LsiModel
    """

    model = mod(corpus_tfidf, id2word=dictionary, num_topics=num_topics)

    vec_bow = dictionary.doc2bow(query.lower().split())
    vec_model = model[vec_bow]  # convert the query to LSI space

    #index these
    index = similarities.MatrixSimilarity(model[corpus_tfidf])

    sims = index[vec_model]  # perform a similarity query against the corpus
    sorted_similarities = sorted(enumerate(sims), key=lambda item: -item[1])

    return sorted_similarities