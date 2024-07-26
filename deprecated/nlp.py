import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models
from collections import defaultdict

# process the data csv
data = pd.read_csv("data/service_reviews_15000rows_translated.csv")

# pull out the 9th column,
# which contains the english translated text, our main corpus
corpus = data.iloc[:, 8]

# preprocess the corpus, rendering it down to lowercase
preprocessed_corpus = []

for i in corpus:
    preprocessed_doc = gensim.utils.simple_preprocess(i, min_len = 2, max_len = 15)
    
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

# turn this into a dictionary structure
dictionary = corpora.Dictionary(processed_corpus)

# create a 'bag of words' corpus using that dictionary
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

# train the model

# tfidf is a transformation that finds term frequency in model frequency
# we will use this in order to create a structure which other models can attack more easily
tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]

# we use an LDA (Latent Dirichlet allocation) model, attempting to classify the documents into 3 clusters
# this is all unsupervised, we have not yet told the model what kind of clusters we are looking for
lda_model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=3)

# we print out the formula. The words show up in order of decreasing coefficient magnitude
# we ideally want the first few words of each formula to be distinct and describe the cluster
lda_model.print_topics(3)



