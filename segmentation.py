
import numpy as np
import json

from parse_sf import TestimonyParser
from textcat import SVMTextcat
from textcat import Vectorizer
from gpt2 import GPT2Scorer
import sys
import spacy
import logging
from summarize import Summarizer

class Segmentor:
    def __init__(self, text, model=None):
        self.segments = [0]  # list of segment starts
        self.segment_spans = []  # list of segment spacy spans
        self.max_srls = []
        self.summaries = []
        self.ps = None  # list of probabilities for the segments
        self.topic_assignments = None  # list of sampled assignments

        if model is not None:
            self.cats = model.topics
        else:
            self.cats = None
        self.model = model
        self.nlp = spacy.load("en_core_web_trf")
        self.parser = TestimonyParser(self.nlp)
        self.summarizer = Summarizer(self.model)
        self.doc = self.parser.parse_testimony(text)  # does CR but not srl
        self.sents = list(self.doc.sents)  # these are spans!!!
        for s in self.sents:
            self.parser.srler.add_to_Span(s, self.parser.srler.parse(s.text))


    def combine_sents(self, window=1, ratio=.5):
        # combines the top sentences. higher ratio means combining more
        # calculates score by GPT2 and given window size
        scorer = GPT2Scorer()
        diffs = []
        for j, s in enumerate(self.sents):
            if j < window or j + window >= len(self.sents):
                continue
            # gpt2_p1 = scorer.sentence_score(" ".join(self.sents[j-window:j+window]))
            gpt2_p1 = scorer.sentence_score(self.doc[self.sents[j-window].start:self.sents[j+window].start].text)
            # gpt2_p2 = scorer.sentence_score(" ".join(self.sents[j-window:j])) \
            #           + scorer.sentence_score(" ".join(self.sents[j:j+window]))
            gpt2_p2 = scorer.sentence_score(self.doc[self.sents[j-window].start:self.sents[j].start].text) \
                      + scorer.sentence_score(self.doc[self.sents[j].start:self.sents[j+window].start].text)
            diffs.append((gpt2_p1 - gpt2_p2, j))

        # diffs.sort(key=lambda x: x[0], reverse=True)
        diffs.sort(reverse=True)
        js = sorted([d[1] for d in diffs[:int(len(diffs) * ratio)]], reverse=True)
        for j in js:
            # self.sents[j-1] = self.sents[j-1] + ' ' + self.sents[j]
            self.sents[j-1] = self.doc[self.sents[j-1].start:self.sents[j].start]
            self.sents[j] = None
        self.sents = [s for s in self.sents if s is not None]
        logging.info("Combined sentences")

    def segment_score(self, start, end):
        # calculate the log-probability for this segment using marginalization
        # returns also the log-probabilities for classification
        # limit the length?
        # single example in batch

        # create span
        span = self.doc[self.sents[start].start:self.sents[end].start]  # we don't want the end sentence too
        # get features
        span._.feature_vector = self.parser.make_new_features(span, bin=int(5 * (start + end) / len(self.sents)))  # the bin is by the middle
        # get probability by model
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

        logging.info(f"len: {len(self.sents)}")
        for i in range(1, len(self.sents)):
            if i % 10 == 0:
                logging.info(f"i: {i}")
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
        # segments = []
        segment_spans = []
        i = len(prev_v) - 1
        p = np.exp(probs[-1])  # this is the probability vector
        ps = [p]
        while i > 0:
            segment_spans = [self.doc[self.sents[prev_v[i]].start:self.sents[i].start]] + segment_spans
            i = prev_v[i]
            # segments = [i] + segments
            if i > 0:
                p = np.exp(probs[i])
                ps = [p] + ps

        # self.segments, self.ps = segments, ps
        self.segment_spans, self.ps = segment_spans, ps
        # return segments, ps
        return segment_spans, ps

    def sample_topics(self, num=3):
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

    def find_srls(self, topic_assignment, count=1):
        # finds srl units for each segment, and ranks them according to the connection with the chosen topic_assignment
        # returns a list (even if count=1)
        # TODO: for now this is only used for the first topic assignment. maybe we can use the assignment distribution??
        for span, topic in zip(self.segment_spans, topic_assignment):
            self.parser.srler.add_to_new_span(span)  # this was probably done already
            # srls, first_last = self.parser.srler.parse_simple(span.text)
            # span._.srls = [span[first:last+1] for first, last in first_last]
            for s in span._.srls:
                s._.feature_vector = self.parser.make_new_features(s, bin=int(5 * (s.start + s.end) / len(self.doc)))
            if span._.srls is not None:
                # should we use priors? should we sample?
                sorted_srls = sorted(zip([self.model.predict(srl)[1][topic] for srl in span._.srls], range(len(span._.srls))), reverse=True)
                self.max_srls.append([span._.srls[i] for _, i in sorted_srls[:count]])
            else:
                self.max_srls.append([])
        return [[srl.text for srl in srls] for srls in self.max_srls]  # this is the texts

    def find_summaries(self, topic_assignment, count=1):
        for span, topic in zip(self.segment_spans, topic_assignment):
            self.summaries.append(self.summarizer.get_ranked_sents(span.as_doc(), max_depth=6, class_num=topic)[:count])  # is topic the same as the label for the model???
        return self.summaries

    def print_segments(self, name=None):
        # segments is list of first sents
        if name is not None:
            f = open(name, "a+")
        # for assignment in self.topic_assignments:
        for i, segment in enumerate(self.segment_spans):
            logging.info(f"************************** Segment {i}, topic: {[self.cats[assignment[i]] for assignment in self.topic_assignments]} ***********************")
            if len(self.max_srls) > 0 and len(self.max_srls[i]) > 0:
                logging.info(f"************************** srls (for first): {[srl.text for srl in self.max_srls[i]]} ***********************")
            elif len(self.summaries[i]) > 0:
                logging.info(f"************************** summaries (for first): {[summary[1] for summary in self.summaries[i]]} ***********************")
            else:
                logging.info(f"************************** {None} ***********************")
            logging.info(segment.text)
            if name is not None:
                f.write(segment.text)
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

def get_sf_testimony_text(i, data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        text = json.load(infile)[str(i)]
    return text

def get_sf_testimony_nums(data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        nums = list(json.load(infile).keys())
    return nums


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import logging.config
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True, })

    # gpt2_window, gpt2_ratio, model_id = sys.argv[1], sys.argv[2], sys.argv[3]
    gpt2_window, gpt2_ratio, model_id = 2, 0.5, "svm"
    logging.info("Starting")
    logging.info(f"GPT2 window: {gpt2_window}")
    logging.info(f"GPT2 ratio: {gpt2_ratio}")
    logging.info(f"model id: {model_id}")
    batch = False
    logging.info(f"Batch: {batch}")
    sys.stdout.flush()

    # model_id = sys.argv[3][:3]
    # if model_id[:3] == "svm":
    #     model_id = "-" + model_id[:3]

    # model = SpacyCat(model_id=model_id)
    model = SVMTextcat(base_path='/cs/snapless/oabend/eitan.wagner/segmentation/').from_path()
    model.find_priors()


    nums = get_sf_testimony_nums()
    # r = range(112, 115)
    r = nums[:3]
    # all_segments = {}
    for i in r:
        logging.info(f'\n\n\nTestimony {i}:')
        # d = Segmentor(text=get_testimony_text(i)[:3000], model=model)
        d = Segmentor(text=get_sf_testimony_text(i)[:], model=model)
        d.combine_sents(window=int(gpt2_window), ratio=float(gpt2_ratio))

        logging.info("\n\nFinding segments: ")
        c = d.find_segments()
        # logging.info(c)
        logging.info("\n\nSampling topics: ")
        assignment = d.sample_topics(num=4)[0]
        # d.find_srls(topic_assignment=assignment, count=2)
        logging.info("\n\nFinding summaries: ")
        d.find_summaries(topic_assignment=assignment, count=4)
        d.print_segments()
        #
        # with open('/cs/snapless/oabend/eitan.wagner/TM_clustering/temp_segments.json', "w+") as outfile:
        #     json.dump(all_segments, outfile)
        #
