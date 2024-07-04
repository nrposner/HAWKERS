import numpy as np
import pandas as pd
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import statistics
from corpus_creation import corpus_maker
from preprocessing import corpus_extractor, preprocess

data = pd.read_csv("data/service_reviews_15000rows_translated.csv")

corpus = corpus_extractor(data)

processed_corpus = preprocess(corpus)

corpus_tfidf, dictionary = corpus_maker(processed_corpus)

def compute_coherence_UMass(corpus, dictionary, k, alpha):
    lsi_model = LsiModel(corpus=corpus_tfidf, num_topics=k)
    coherence = CoherenceModel(model=lsi_model,
                              corpus=corpus,
                              dictionary=dictionary,
                              coherence='u_mass')
    return coherence.get_coherence() + alpha*k
coherenceList_UMass = []
numTopicsList = np.arange(3, 10, 1)
for k in numTopicsList:
    c_UMass = compute_coherence_UMass(corpus_tfidf, dictionary, k, 0)
    coherenceList_UMass.append(c_UMass)



def topic_num_selector():
    choice = []
    for i in np.arange(50):
        coherenceList_UMass = []
        numTopicsList = np.arange(3, 10, 1)
        for k in numTopicsList:
            c_UMass = compute_coherence_UMass(corpus_tfidf, dictionary, k, 0)
            coherenceList_UMass.append(c_UMass)
        choice.append(coherenceList_UMass.index(min(coherenceList_UMass)))
    return statistics.mode(choice)