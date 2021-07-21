
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from scipy.special import log_softmax
import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson
# from torch import log_softmax

from transformers import AutoModelWithLMHead, AutoTokenizer
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

from spacy.tokens import Token
Token.set_extension("depth", default=None, force=True)


class Summarizer:
    def __init__(self, classifier):
        self.model = AutoModelWithLMHead.from_pretrained("t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.classifier = classifier

    def sents_score(self, texts):
        # returns scores for given texts this should be a 1-dim array
        inputs1 = self.tokenizer(["summarize: " for _ in texts], return_tensors='pt', padding=True)
        inputs1 = inputs1.to(dev)
        inputs2 = self.tokenizer([text for text in texts], return_tensors='pt', padding=True)
        inputs2 = inputs2.to(dev)

        out = self.model(input_ids=inputs1['input_ids'], labels=inputs2['input_ids'])['logits']
        ln = inputs2['input_ids'][0].detach().cpu().numpy()

        sm = log_softmax(out.detach().cpu().numpy(), axis=2)  # check dimensions!!
        # this is P(x|X)
        log_summary_probs = np.sum(sm[:, range(1, sm.shape[1]-1), ln[1:-1]], axis=1)  # check that we need this also for t5

        # use classifier score. this is logP(t|x).
        log_class_probs = [self.classifier.predict_raw(text) for text in texts]  # this needs a span!!! make a function for text (and bin?) only??
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

    def make_sents(self, doc, max_depth):
        # makes list of new sentences for the whole doc with up to depth max_depth
        def _make_sent(sent, depth):
            return " ".join([t.text for t in sent if t._.depth <= depth])

        new_sents = []
        for sent in doc.sents:
            for m_d in range(max_depth):
                new_sents.append(_make_sent(sent, m_d))

    def get_ranked_sents(self, doc, max_depth):
        # returns a list of tuples tuples (log_prob,text)
        new_sents = self.make_sents(doc, max_depth)
        log_probs = self.sents_score(new_sents)

        return sorted(list(zip(log_probs, new_sents)))
