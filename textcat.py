from numpy import vstack
from numpy import argmax
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import joblib

from tqdm import tqdm

import spacy
# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class TextcatSVM:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def from_path(self, model_path='svm.pkl', vectorizer_path='tfidf.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def fit(self, data_path=None):
        dataset = TextcatDataset(data_path)
        self.vectorizer = dataset.vectorizer
        self.model= train_svm(dataset, random_state=42)

    def predict(self, span):
        # returns probabilities for a Span
        x = self.vectorizer.transform(span[0])
        return self.model.predict_proba(x)


class Vectorizer:
    # creates vectors for texts
    # the text should be spaCy Spans
    def __init__(self):
        self.tfidf = TfidfVectorizer()

    def fit_transform(self, spans):
        texts = [s.text for s in spans]
        X = self.tfidf.fit_transform(texts)
        # here we add features
        return X

    def transform(self, spans):
        texts = [s.text for s in spans]
        X = self.tfidf.transform(texts)
        # here we add features
        return X


# dataset definition
class TextcatDataset:
    # hold the dataset and performs transforming and splitting
    def __init__(self, path):
        # load the data
        with open(path, 'r') as infile:
            data = json.load(infile)

        np.random.shuffle(data)
        # store the inputs and outputs
        self.y, self.X = list(zip(*data))
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.X)
        self.X = csr_matrix(self.X, dtype=np.float32)

        # label encode target and ensure the values are floats
        self.encoder = LabelEncoder().fit(self.y)
        self.y = self.encoder.transform(self.y)

        self.X_train = self.X[:int(0.8*self.X.shape[0]), :]
        self.X_test = self.X[int(0.8*self.X.shape[0]):, :]
        self.y_train = self.y[:int(0.8*len(self.y))]
        self.y_test = self.y[int(0.8*len(self.y)):]

        self.dim = self.X.shape[1]

    # number of rows in the dataset
    def __len__(self):
        return self.X.shape[0]

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2, random_state=42):
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=random_state)
        # return X_train, X_test, y_train, y_test
        return self.X_train, self.X_test, self.y_train, self.y_test

    def id2lables(self, y):
        # converts a response vector back to the original label strings
        return self.encoder.inverse_transform(self.y)


def train_svm(dataset, random_state=42, out_path=None):
    # trains an SVM model on the given dataset
    X_train, X_test, y_train, y_test = dataset.get_splits(0.2, random_state=random_state)

    from sklearn import svm
    #Create a svm Classifier
    clf = svm.SVC(kernel='rbf', probability=True, gamma='scale', class_weight='balanced')
    #Train the model using the traini]
    # ng sets
    print("Training SVM")
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    #Import scikit-learn metrics module for accuracy calculation
    # Model Accuracy: how often is the classifier correct?
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
    # now you can save it to a file
    if out_path:
        joblib.dump(clf, out_path + 'svm.pkl')
        joblib.dump(dataset.vectorizer, out_path + 'tfidf.pkl')
    return clf


def test_model(model, dataset, random_state=42, out_path=None):
    X_train, X_test, y_train, y_test = dataset.get_splits(0.2, random_state=random_state)
    labels = dataset.encoder.classes_
    import pandas as pd

    train_conf = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels)
    eval_conf = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels)

    # train_texts = list(zip(X_train, dataset.encoder.inverse_transform(y_train)))
    # eval_texts = list(zip(X_test, dataset.encoder.inverse_transform(y_test)))
    y_labels_train = dataset.encoder.inverse_transform(y_train)
    y_labels_eval = dataset.encoder.inverse_transform(y_test)
    train_correct = 0
    eval_correct = 0
    # column is the predicted and row is the real. NO!!! the opposite!
    preds = dataset.encoder.inverse_transform(model.predict(X_train))
    for i, p in enumerate(preds):
        train_conf[y_labels_train[i]][p] += 1

    preds = dataset.encoder.inverse_transform(model.predict(X_test))
    for i, p in enumerate(preds):
        eval_conf[y_labels_eval[i]][p] += 1

    if out_path:
        train_conf.to_csv(out_path + f"models/train_conf_svm.csv")
        eval_conf.to_csv(out_path + f"models/eval_conf_svm.csv")


if __name__ == "__main__":
    # prepare the data
    # path = 'sf_title_w_segments1.json'
    path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/sf_title_w_segments1.json'
    dataset = TextcatDataset(path)
    # model = train_svm(dataset, random_state=42, out_path='/cs/snapless/oabend/eitan.wagner/segmentation/models/')
    # now you can save it to a file
    model = joblib.load('/cs/snapless/oabend/eitan.wagner/segmentation/models/svm.pkl')
    test_model(model, dataset, random_state=42, out_path='/cs/snapless/oabend/eitan.wagner/segmentation/models/')
