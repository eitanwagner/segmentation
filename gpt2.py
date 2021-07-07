import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from scipy.special import log_softmax
import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson
# from torch import log_softmax

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

# from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# MAX_LEN, MIN_LEN = 1000, 35
MAX_LEN, MIN_LEN = 800, 35


class GPT2Scorer:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.to(dev)

    def sentence_score(self, sent):
        # returns sentence probability (in log space)
        inputs = self.tokenizer(sent, return_tensors="pt")
        # print(inputs['input_ids'].shape)
        inputs = inputs.to(dev)

        return self.model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])['loss'].item() * inputs['input_ids'].shape[1]

        # check why this doesn't work!!
        # out = self.model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])['logits']
        # print(out.shape)
        # ln = inputs['input_ids'][0].detach().cpu().numpy()
        # sm = log_softmax(out[0].detach().cpu().numpy(), axis=0)
        # print(sm.shape)
        # return sum(sm[range(sm.shape[0]), ln])
        # return sum(sm[range(1, sm.shape[0] - 1), ln[1:-1]])

class GPT2wDMM(GPT2Scorer):
    def __init__(self, TM=None):
        super().__init__()
        self.TM = TM

    def score_with_topic(self, topic_model, sent, log=True):
        # returns sentence probability (in log space)
        inputs = self.tokenizer(sent, return_tensors="pt")
        # print(inputs['input_ids'].shape)
        inputs = inputs.to(dev)
        logits = self.model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])['logits'][0].detach().cpu().numpy()

        w2lemmatized = self.TM.lemmatize(sent) # should return a dict from word to the lemmatized word, with None for out of vocab words
        tm_logps = self.TM.probabilities(list(w2lemmatized.values()))  # should return prior log probabilites for words with None
        for i, w in enumerate(list(w2lemmatized.keys())):
            w2t = inputs.word_to_tokens(list(w2lemmatized.values()))
            w_logits = np.sum(logits[:, w2t], axis=1)



if __name__ == "__main__":
    pass