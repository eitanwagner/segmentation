
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from scipy.special import log_softmax
import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson
# from torch import log_softmax
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging


if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

from spacy.tokens import Token
Token.set_extension("depth", default=None, force=True)


class Summarizer:
    def __init__(self, classifier, default_len=10):
        self.default_len = default_len  # this is measured in tokens and not words
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.model.to(dev)
        # self.tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.classifier = classifier

    def sents_score(self, texts, class_num):
        # returns scores for given texts this should be a 1-dim array
        # logging.info(f"len(texts): {len(texts)}")
        inputs1 = self.tokenizer(["summarize: " for _ in texts], return_tensors='pt', padding=True)
        inputs1 = inputs1.to(dev)
        inputs2 = self.tokenizer([text for text in texts], return_tensors='pt', padding=True)
        inputs2 = inputs2.to(dev)
        # logging.info(f"inputs1: {inputs1['input_ids'][0].size()}")
        # logging.info(f"inputs2: {inputs2['input_ids'][0].size()}")

        out = self.model(input_ids=inputs1['input_ids'], labels=inputs2['input_ids'])['logits']
        ln = inputs2['input_ids'][0].detach().cpu().numpy()
        # logging.info(f"ln.shape: {ln.shape}")

        sm = log_softmax(out.detach().cpu().numpy(), axis=2)  # check dimensions!!
        # logging.info(f"sm.shape: {sm.shape}")
        # this is P(x|X)
        log_summary_probs = np.mean(sm[:, range(1, sm.shape[1]-1), ln[1:-1]], axis=1)  # using mean log-probability
        log_summary_probs += poisson.logpmf(len(inputs2), self.default_len * np.ones(len(texts)))  # !!!!
        # logging.info(f"log summary probs.shape: {log_summary_probs.shape}")

        # use classifier score. this is logP(t|x).
        log_class_probs = np.array([self.classifier.predict_raw(text)[class_num] for text in texts])
        # logging.info(f"log_class_probs.shape: {log_class_probs.shape}")
        # Assuming this does not depent on X, then the sum is logP(t,x|X) and assuming uniform P(t) the subtractions is logP(x|t,X)
        return log_summary_probs + log_class_probs - np.log(len(self.classifier.topics))

    def add_depth(self, doc):
        # add ._.depth property to every token in the doc
        def _add_depth_recursive(node, depth):
            node._.depth = depth
            if node.n_lefts + node.n_rights > 0:
                return [_add_depth_recursive(child, depth + 1) for child in node.children]

        for sent in doc.sents:
            _add_depth_recursive(sent.root, 0)

    def make_random(self, sent, depth):
        words = []
        choices = np.random.uniform(size=len(sent)) > .2
        def _add_recursive(node):
            if node._.depth <= depth:
                words.append(node)
            if node.n_lefts + node.n_rights > 0:
                [_add_recursive(child) for i, child in enumerate(node.children) if choices[i]]
        for _ in range(100):
            _add_recursive(sent.root)
            if len(words) > 2 and len(sent) > 2:
                return " ".join([t.text for t in sent if t in words])
            words = []

    def make_sents(self, doc, max_depth, random=10):
        # makes list of new sentences for the whole doc with up to depth max_depth
        # random is the number of random samples for each sentence
        def _make_sent(sent, depth):
            return " ".join([t.text for t in sent if t._.depth <= depth])

        new_sents = []
        for sent in doc.sents:
            if random is None:
                for m_d in range(1, max_depth):  # not just the root
                    if len(sent) > 3:  # not too short
                        new_sents.append(_make_sent(sent, m_d))
            else:
                for _ in range(random):
                    new_sents.append(self.make_random(sent, depth=max_depth))
        if len(new_sents) > 0:
            s = set(new_sents)
            s.discard(None)
            return list(s)
        else:
            return []



    def get_ranked_sents(self, doc, max_depth, class_num):
        # returns a list of tuples tuples (log_prob,text)
        self.add_depth(doc)
        new_sents = self.make_sents(doc, max_depth)
        if len(new_sents) == 0:
            return [(1, "")]
        # print(new_sents)
        log_probs = self.sents_score(new_sents, class_num)

        return sorted(list(zip(log_probs, new_sents)), reverse=True)
