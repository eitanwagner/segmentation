
import numpy as np
import json

from parse_sf import TestimonyParser
from textcat import SVMTextcat
from gpt2 import GPT2Scorer
import sys
import spacy

class Segmentor:
    def __init__(self, text, model=None):
        self.segments = [0]  # list of segment starts
        self.ps = None  # list of probabilities for the segments
        self.topic_assignments = None  # list of sampled assignments

        self.cats = model.topics
        self.model = model
        self.nlp = spacy.load("en_core_web_trf")
        self.sents = list(self.doc.sents)  # these are spans!!!
        self.parser = TestimonyParser(self.nlp)
        self.doc = self.parser.parse_testimony(text)

    def combine_sents(self, window=1, ratio=.5):
        # combines the top sentences. higher ratio means combining more
        # calculates score by GPT2 and given window size
        scorer = GPT2Scorer()
        diffs = []
        for j, s in enumerate(self.sents):
            if j < window:
                continue
            # gpt2_p1 = scorer.sentence_score(" ".join(self.sents[j-window:j+window]))
            gpt2_p1 = scorer.sentence_score(self.doc[self.sents[j-window].start:self.sents[j+window].end].text)
            # gpt2_p2 = scorer.sentence_score(" ".join(self.sents[j-window:j])) \
            #           + scorer.sentence_score(" ".join(self.sents[j:j+window]))
            gpt2_p2 = scorer.sentence_score(self.doc[self.sents[j-window].start:self.sents[j].end].text) \
                      + scorer.sentence_score(self.doc[self.sents[j].start:self.sents[j+window].end].text)
            diffs.append((gpt2_p1 - gpt2_p2, j))

        # diffs.sort(key=lambda x: x[0], reverse=True)
        diffs.sort(reverse=True)
        js = sorted([d[1] for d in diffs[:int(len(diffs) * ratio)]], reverse=True)
        for j in js:
            # self.sents[j-1] = self.sents[j-1] + ' ' + self.sents[j]
            self.sents[j-1] = self.doc[self.sents[j-1].start:self.sents[j].end]
            self.sents[j] = None
        self.sents = [s for s in self.sents if s is not None]

    def segment_score(self, start, end):
        # calculate the log-probability for this segment using marginalization
        # returns also the log-probabilities for classification
        # limit the length?
        # single example in batch
        span = self.doc[self.sents[start].start:self.sents[end].end]  # is the .end token included???
        span._.feature_vector = make_features(span, i=5 * (start + end) // len(self.sents), nlp=self.nlp)  # the bin is by the middle

        p, probs = self.model.predict(span)
        return p, probs

    def find_segments(self, use_heuristic=True):
        """
        :param use_heuristic:
        :param sents: list of sentences
        :return: list of firsts in segments (always starting from 0), and list of states
        """
        # we need a prior for the segment number?
        prev_v = [0]  # a list of previous vertices. so 0 will also have a predecessor
        probs = [None]  # a list of probability vectors for each topic with the optimal prev_v
        prev_score = [0]  # a list of scores with the optimal previous
        last = 0  # for heuristic. maybe relax to use previous last

        print("len: ", len(self.sents))
        for i in range(1, len(self.sents)):
            if i % 10 == 0:
                print("i: ", i)
                sys.stdout.flush()

            js = range(last, i)
            # find previous nodes with respective scores
            prevs = [(prev_score[j], ) + self.segment_score(j, i) for j in js]
            p_costs = [p[0] + p[1] for p in prevs]

            # find best prev
            # prevent same segment??
            m = int(np.argmax(p_costs))
            prev_score.append(p_costs[m])
            # print("prev_state:", prev_state)
            probs.append(prevs[m][2])  # this should be a vector

            prev_v.append(m + last)
            if use_heuristic:
                last += m

        # backward pass
        segments = []
        i = len(prev_v) - 1
        p = np.exp(probs[-1])  # this is the probability vector
        ps = [p]
        while i > 0:
            i = prev_v[i]
            segments = [i] + segments
            if i > 0:
                p = np.exp(probs[i])
                ps = [p] + ps

        self.segments, self.ps = segments, ps
        return segments, ps

    def sample_topics(self, num=1):
        # returns random topic assignments, without doubles, for the requested amount
        def has_doubles(l):
            for i, v in enumerate(l[1:], start=1):
                if v == l[i-1]:
                    return True
            return False

        attempts = 0
        found = 0
        topic_assignments = []
        while attempts < 100:
            topics = [np.random.choice(a=len(p), p=p/p.sum()) for p in self.ps]
            if not has_doubles(topics):
                found += 1
                topic_assignments.append(topics)
                if found == num:
                    self.topic_assignments = topic_assignments
                    return topic_assignments
        return None

    def srls(self, topic_assignment):
        # finds srl units for each segement, and ranks them according to the connection with the chosen topic_assignment
        return

    def print_segments(self, name=None):
        # segments is list of first sents
        if name is not None:
            f = open(name, "a+")
        with_last = self.segments + [len(self.sents)]
        for i in range(len(self.segments)):
            print(f"************************** Segment {i}, topic for each assignment: {[self.cats[s[i]] for s in self.topic_assignments]} ***********************")
            for s in self.sents[with_last[i]:with_last[i+1]]:
                print(s)
                if name is not None:
                    f.write(s)
        if name is not None:
            f.close()


def get_testimony_sents(i, data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    with open(data_path + 'sents.json', 'r') as infile:
        sents = json.load(infile)[str(i)]
    return sents

def get_testimony_text(i, data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    with open(data_path + 'raw_text.json', 'r') as infile:
        text = json.load(infile)[str(i)]
    return text

if __name__ == '__main__':
    print("Starting")
    print("GPT2 window:", sys.argv[1])
    print("GPT2 ratio:", sys.argv[2])
    print("model id:", sys.argv[3])
    batch = False
    print("Batch:", batch)
    sys.stdout.flush()

    # model_id = sys.argv[3][:3]
    # if model_id[:3] == "svm":
    #     model_id = "-" + model_id[:3]

    # model = SpacyCat(model_id=model_id)
    model = SVMTextcat(base_path='/cs/snapless/oabend/eitan.wagner/segmentation/').from_path()
    model.find_priors()

    # all_segments = {}
    for i in range(111, 114):
        print(f'\n\n\nTestimony {i}:')
        d = Segmentor(text=get_testimony_text(i)[:], model=model)
        d.combine_sents(window=int(sys.argv[1]), ratio=float(sys.argv[2]))

        print("\n\nFinding segments: ")
        c = d.find_segments()
        print(c)
        d.print_segments()
        #
        # with open('/cs/snapless/oabend/eitan.wagner/TM_clustering/temp_segments.json', "w+") as outfile:
        #     json.dump(all_segments, outfile)
        #
