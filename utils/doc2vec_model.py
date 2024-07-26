import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def d2v_classifier(train_docs:np.ndarray, classes:np.ndarray, test_docs:np.ndarray, topn:int = 5, workers=1, vector_size = 4, epochs=20, dm=1):
    """Classify documents according to desired classes. 
    
    train_docs and test_docs must contain only the text of the documents, while
    classes contains the tags for the training documents. 
    
    """
    
    train_data = pd.DataFrame({"docs": train_docs, "tag":classes})
    train_tagged = train_data.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['docs']), tags=[r.tag]), axis=1)
    
    trainsent = train_tagged.values
    
    doc2vec_model = Doc2Vec(trainsent, workers=workers, vector_size=vector_size, epochs=epochs, dm=dm)
    
    train_targets, train_regressors = zip(
        *[(doc.tags[0], doc2vec_model.infer_vector(doc.words)) for doc in trainsent])
    
    inferred_test_vectors = [doc2vec_model.infer_vector(doc.split()) for doc in test_docs]
    
    knn_test_predictions = [
        doc2vec_model.dv.most_similar([pred_vec], topn=topn)[0][0]
        for pred_vec in inferred_test_vectors
    ]
    
    predictions = pd.DataFrame({"doc":test_docs, "class": knn_test_predictions})
    
    return predictions

def doc2vec_viz(y_test, preds):
    labels = ['Customer Service', 'Delivery Service', 'Mixed', 'Other', 'Product Quality',]

    conf_mat = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Heatmap of Doc2Vec Classification")

def doc2vec_pipeline(tagged_data, untagged_data, topn:int = 5, workers=1, vector_size = 4, epochs=20, dm=1):
    train_docs = tagged_data["DS_TEXT_TRANSLATED"]
    classes = tagged_data["Tags"]
    test_docs = untagged_data["DS_TEXT_TRANSLATED"]
    
    preds = d2v_classifier(train_docs, classes, test_docs, topn, workers, vector_size, epochs, dm)
    
    return preds
    
    