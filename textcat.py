from numpy import vstack
from numpy import argmax
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
from scipy.stats import poisson
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import joblib

from tqdm import tqdm

import spacy
from spacy.vocab import Vocab
nlp = spacy.load("en_core_web_trf")


class SVMTextcat:
    def __init__(self, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/'):
        self.model = None
        self.vectorizer = None
        self.base_path = base_path
        self.topics = None
        self.prior_length = None

    def from_path(self, model_path='models/svm.pkl', vectorizer_path='models/vectorizer.pkl'):
        self.model = joblib.load(self.base_path + model_path)
        self.vectorizer = joblib.load(self.base_path + vectorizer_path)

        self.topics = self.model.classes_  # do we need to transform this??
        return self

    def fit(self, data_path=None, test=False):
        dataset = TextcatDataset(self.base_path + data_path)
        self.vectorizer = dataset.vectorizer
        self.model = train_svm(dataset, random_state=42, out_path=self.base_path+"models/")  # !!!
        test_model(self.model, dataset, out_path=self.base_path + 'models/results/')

        self.topics = self.vectorizer.encoder.classes_  # these are the strings
        return self

    def predict(self, span):
        # returns probabilities for a Span, as a tuple of (marginal_likelihood, classification_probability)
        x = self.vectorizer.transform(span)

        # do we need P(t)?
        pred_logp = self.model.predict_log_proba(x)  # logP(t|x,s)
        vocab_size = len(self.vectorizer.tfidf.vocabulary_)
        lm_logp = -len(span) * np.log(vocab_size)  # logP(x|s)
        len_logp = poisson.logpmf(len(doc)//8, self.prior_length//8)  # logP(s)
        return logsumexp(preds_logp + lm_logp + len_logp), log_preds

    def find_priors(self):
        # TODO: use new data. Or at least calculate real prior!!
        with open(self.base_path + 'data/title_w_segments.json', 'r') as infile:
            title_w_segments = json.load(infile)

        lengths = [len(text.split(" ")) for _, text in title_w_segments]
        self.prior_length = sum(lengths) / len(title_w_segments)  # the average length. not exact!!!
        print("calculated priors")


class Vectorizer:
    # creates vectors for texts
    # the text should be spaCy Spans
    # this also deals with the labels
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.encoder = LabelEncoder()

    def fit_transform(self, spans):
        texts = [s.text for s in spans]
        extra_features = [s._.feature_vectors for s in spans]  # any span should have this
        X = self.tfidf.fit_transform(texts)
        X = np.concatenate((X, np.array(extra_features).T), axis=1)
        # here we add features
        return X

    def transform(self, spans):
        texts = [s.text for s in spans]
        extra_features = [s._.feature_vectors for s in spans]  # any span should have this
        X = self.tfidf.transform(texts)
        # here we add features
        X = np.concatenate((X, np.array(extra_features).T), axis=1)
        return X

    def label_fit(self, y):
        # learns label encoding (from string to id)
        self.encoder.fit(y)
        return self

    def label_transform(self, y):
        # transform string to id
        return self.encoder.transform(y)

    def label_inverse_transform(self, y):
        # converts a response vector back to the original label strings
        return self.encoder.inverse_transform(y)


# dataset definition
class TextcatDataset:
    # hold the dataset and performs transforming and splitting
    def __init__(self, path):
        # load the data
        # with open(path, 'r') as infile:
        #     data = json.load(infile)
        vocab = Vocab().from_disk(path + "vocab")
        doc_bin = DocBin().from_disk(path + "data.spacy")

        data = [(segment, segment._.real_topic) for doc in doc_bin.get_docs(nlp.vocab) for segment in doc]
        np.random.shuffle(data)
        # store the inputs and outputs
        spans, self.y = list(zip(*data))
        # self.y, self.X = list(zip(*data))
        # self.vectorizer = TfidfVectorizer()
        self.vectorizer = Vectorizer()
        self.X = self.vectorizer.fit_transform(spans)
        self.X = csr_matrix(self.X, dtype=np.float32)

        # label encode target and ensure the values are floats
        self.vectorizer.label_fit(self.y)
        self.y = self.vectorizer.label_transform(self.y)

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
    clf = svm.SVC(kernel='rbf', probability=True, gamma='scale', class_weight='balanced')
    # Train the model using the training sets
    print("Training SVM")
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
    # save
    if out_path:
        joblib.dump(clf, out_path + 'svm.pkl')
        joblib.dump(dataset.vectorizer, out_path + 'vectorizer.pkl')
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
        train_conf.to_csv(out_path + f"train_conf_svm.csv")
        eval_conf.to_csv(out_path + f"eval_conf_svm.csv")


if __name__ == "__main__":
    path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/'
    model = SVMTextcat(base_path=path)
    model.fit(data_path='data/', test=True)
    #
    # dataset = TextcatDataset(path)
    # model = train_svm(dataset, random_state=42, out_path='/cs/snapless/oabend/eitan.wagner/segmentation/models/')
    # # now you can save it to a file
    # # model = joblib.load('/cs/snapless/oabend/eitan.wagner/segmentation/models/svm.pkl')
    # test_model(model, dataset, random_state=42, out_path='/cs/snapless/oabend/eitan.wagner/segmentation/models/')
