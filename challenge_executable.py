import os
import string
import random
from collections import Counter

    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.exceptions import ConvergenceWarning

import warnings


THRESHOLD = 0.8 ## confidence we want in the probability that the message is about a lost baggage 

## Loading the dataset 


data = pd.read_csv("intent-detection-train.csv" ,sep = ',')


# Je définie une graine qui permettra de reproduire les resultats 
seed = 39
random.seed(seed)

data_set = data.copy()
# Transformation des labels en valeur numérique pour les modèles 


data_set["label"] = data["label"].replace(to_replace={"translate" : 0,
                                                 "travel_alert" : 1,
                                                 "flight_status" : 2,
                                                 "lost_luggage" : 3,
                                                 "travel_suggestion" : 4,
                                                 "carry_on" : 5,
                                                 "book_hotel" : 6,
                                                 "book_flight" : 7,
                                                 "out_of_scope":8})


print(f"Ce dataset contient : {data_set.shape[0]} elements, voici 5 élements tirés au hasard :  {data_set.sample(5)}")
print(f"Distribution des labels: {data.label.value_counts(normalize=True)}")


# Je sépare le dataset afin de pouvoir tester mon classifier après l'entrainement de mes modèles, une répartition de 80/20 
# a été choisie 



x_train, x_test, y_train, y_test = train_test_split(data_set['text'].values, data_set['label'].values, test_size=0.15, random_state=seed)


print("Size of train set is: ", len(y_train))
print("Size of test set is: ", len(y_test))


# Le choix du v

classifier_dict = {
    "bagofwords+multiNB": Pipeline([('vectorizer', CountVectorizer()), ('classifier', MultinomialNB(alpha=0.2))]),
    "stopwords+tfidf+multiNB": Pipeline([('vectorizer', TfidfVectorizer(stop_words='english')), ('classifier', MultinomialNB(alpha=0.2))]),
    "bagofwords+multiLogistic": Pipeline([('vectorizer', CountVectorizer()), ('classifier', LogisticRegression(solver="saga" , multi_class="auto", max_iter=200))]),
}


def apply_thresh_lost(prob_vector, threshold):
    # Check if the probability of the fourth class is below the threshold
    sorted = np.argsort(prob_vector)
    #print(sorted[-1])
    if (prob_vector[3] < threshold and prob_vector[3] == sorted[-1]):
        # Choose the second highest probability as the prediction
        return np.argsort(prob_vector)[-2]
    else:
        return np.argmax(prob_vector)
        



def train(
    classifier, 
    X_train, 
    y_train, 
    rnd_state_input , 
    test_split_size=0.2, 
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=test_split_size,
            random_state=rnd_state_input
        )
        classifier.fit(X_train, y_train)
        
        y_proba = classifier.predict_proba(X_val)
        #print(y_proba)
        y_pred = [apply_thresh_lost(prob_vector, THRESHOLD) for prob_vector in y_proba]
        #print(y_pred)
        print(y_pred,y_val, classifier.score(X_val, y_val))
        if rnd_state_input == 1:
            print(f"The accuracy score for this turn is : {accuracy_score(y_val, y_pred)}")
            print(f"The precision score for this turn is : {precision_score(y_val, y_pred, average='macro')}")
            print(f"The recall score for this turn is : {recall_score(y_val, y_pred, average='macro', zero_division= True)}")
            print(f"The f1_score score for this turn is : {f1_score(y_val, y_pred, average='macro')}")
            
        # print("\t|| k=5 Accuracy: {}% ".format(accuracy_score(y_val, y_pred)))
        # print("\t|| k=5 Precision: {}% ".format(precision_score(y_val, y_pred, average='macro')))
        # print("\t|| k=5 Recall: {}% ".format(recall_score(y_val, y_pred, average='macro')))
        # print("\t|| k=5 F1: {}% ".format(f1_score(y_val, y_pred, average='macro')))
        return classifier, classifier.score(X_val, y_val)
 
 
def plot_confusion_matrix(classifier, X_test, y_test, labels):
    y_pred = classifier.predict(X_test)
    print(y_pred.T , y_test.T)
    confusion_mat = confusion_matrix(y_test, y_pred)
    confusion_mat = normalize(confusion_mat , axis=1 , norm='l1' )
    # Plot confusion_matrix
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(confusion_mat, annot=True, cmap = "flare", fmt ="0.2f", xticklabels=labels, yticklabels=labels)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
   
train_validation_random_state = [1,5,10,15,20]



for model_name, model in classifier_dict.items():
    print("~~~~~~~~~~~~~~~~~~~~")
    print(model_name + " : ") 
    all_cross_val_scores = []
    for k in train_validation_random_state:
        classifier, score = train(
            classifier=model, 
            X_train=x_train, 
            y_train=y_train, 
            rnd_state_input=k
        )
        all_cross_val_scores.append(score)
    print(all_cross_val_scores)
    all_cross_val_scores_np = np.array(all_cross_val_scores)
    mean_score = all_cross_val_scores_np.mean()
    print("Mean accuracy score on news: ", mean_score)
    plot_confusion_matrix(classifier, x_test, y_test, data["label"].unique())