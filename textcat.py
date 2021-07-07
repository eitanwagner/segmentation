# pytorch mlp for multiclass classification
from numpy import vstack
from numpy import argmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import random_split

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# dataset definition
class TextcatDataset(Dataset):
    # load the dataset
    def __init__(self, path, tfidf=True, sparse=True):
        # load the data
        with open(path, 'r') as infile:
            data = json.load(infile)

        # store the inputs and outputs
        self.y, self.X = list(zip(*data))
        if tfidf:
            vectorizer = TfidfVectorizer()
            self.X = vectorizer.fit_transform(self.X).toarray()
        # ensure input data is floats
        if sparse:
            from scipy.sparse import csr_matrix
            self.X = csr_matrix(self.X, dtype=np.float32)
        else:
            self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.encoder = LabelEncoder().fit(self.y)
        self.y = self.encoder.transform(self.y)
        self.dim = self.X.shape[1]

    # number of rows in the dataset
    def __len__(self):
        return self.X.shape[0]

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        # determine sizes
        test_size = round(n_test * self.X.shape[0])
        train_size = self.X.shape[0] - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


def with_svm(path, only_data=False):
    # load the dataset
    dataset = TextcatDataset(path, tfidf=True)
    # calculate split
    X, y = dataset.X, dataset.y
    X_train = X[:int(0.8*X.shape[0]), :]
    X_test = X[int(0.8*X.shape[0]):, :]
    y_train = y[:int(0.8*len(y))]
    y_test = y[int(0.8*len(y)):]
    z = list(zip(X_train, y_train))
    np.random.shuffle(z)
    X_train, y_train = list(zip(*z))
    from scipy.sparse import vstack
    X_train = vstack(X_train)
    if only_data:
        return (dataset, X_train, y_train, X_test, y_test)
    # from scipy.sparse import csr_matrix
    # # X_train = csr_matrix(np.array(X_train))
    # X_train = csr_matrix(X_train)
    #Import svm model
    from sklearn import svm
    #Create a svm Classifier
    clf = svm.SVC(kernel='rbf', probability=True)
    #Train the model using the training sets
    print("Training SVM")
    clf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy: how often is the classifier correct?
    print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))
    from sklearn.externals import joblib
    # now you can save it to a file
    joblib.dump(clf, 'svm.pkl')
    return clf, (dataset, X_train, y_train, X_test, y_test)

# evaluate the model
def evaluate_model(model):
    predictions, actuals = list(), list()
    model.eval()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

def test_model(model, data):
    dataset, X_train, y_train, X_test, y_test = data
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

    train_conf.to_csv(f"train_conf_svm.csv")
    eval_conf.to_csv(f"eval_conf_svm.csv")


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


# prepare the data
path = 'sf_title_w_segments1.json'
# path = '/cs/snapless/oabend/eitan.wagner/TM_clustering/title_w_segments.json'
# model, data = with_svm(path)
data = with_svm(path, only_data=True)
from sklearn.externals import joblib
# now you can save it to a file
model = joblib.load('svm.pkl')
test_model(model, data)



