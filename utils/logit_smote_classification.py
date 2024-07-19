import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def logit_smote_classification(df):
    #initializing data source
    before_vect = df[["DS_TEXT_TRANSLATED", "Tags"]]

    #creating column transformer
    columnTransformer = ColumnTransformer([('E',OneHotEncoder(dtype='int'),["DS_TEXT_TRANSLATED", "Tags"]),
                                           ('tfidf',TfidfVectorizer(stop_words=None, max_features=100000), 
                                            'DS_TEXT_TRANSLATED')], remainder='drop')

    #transforming the dataset into its vectorized form
    vector_transformer = columnTransformer.fit(before_vect)
    vectorized_df = vector_transformer.transform(before_vect)

    y = before_vect["Tags"]
    y=y.to_frame()

    #gathering the training and testing set. Note that there is no y_test variable, 
    #because we do not have classifications for the majority of this dataset. 
    #a version of this operating only on the classified data performed very well on testing

    X_train = vectorized_df[before_vect["Tags"].isna()==False]

    X_test = vectorized_df[before_vect["Tags"].isna()==True]

    Y_train = y[before_vect["Tags"].isna()==False]


    #creating the synthetic elements. Note we create these only from the training
    #data, because creating synthetic data from both train and test would
    #create a data leak, corrupting the integrity of our train/test division
    smote = SMOTE(random_state=777,k_neighbors=5)
    X_smote,y_smote = smote.fit_resample(X_train, Y_train)

    #model creation
    model = LogisticRegression()

    #we fit the model on the mix of real and synthetic training data
    model.fit(X_smote, y_smote)

    #we predict the classifications
    y_pred = model.predict(X_test)

    tests = before_vect[before_vect["Tags"].isna()==True]["DS_TEXT_TRANSLATED"]

    tests_df = pd.DataFrame(data={"text":tests, "classification": y_pred})

    return tests_df

