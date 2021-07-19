
import logging
import json
# import pickle
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

from segment_srl import Referencer, SRLer
import segment_srl

import pandas as pd
import numpy as np
import spacy
from spacy.tokens import Span
from spacy.tokens import DocBin
from spacy.tokens import Token

Span.set_extension("feature_vector", default=None)  # type list
Span.set_extension("real_topic", default=None)  # type String

span_extensions = ["bin"]  # for removing
token_extensions = ["segment"]

# VECTOR_LEN = 100
VECTOR_LEN = 768

def add_extensions():
    for ext in span_extensions:
        Span.set_extension(ext, default=None)
    for ext in token_extensions:
        Token.set_extension(ext, default=None)
    segment_srl.add_extensions()

def remove_extensions():
    for ext in span_extensions:
        Span.remove_extension(ext)
    for ext in token_extensions:
        Token.remove_extension(ext)
    segment_srl.remove_extensions()


# not used
def count_topics():
    data = pd.read_csv("Martha_transcripts/index segments for 1000 English Jewish survivor interviews.csv")
    # print(data.head())

    testimonies = set(data['IntCode'])
    topics = [t for d in data['IndexedTermLabels'] for t in str(d).split("; ")]
    print(len(topics))

    with open('words2topics.json', 'r') as infile:
        words2topics = json.load(infile)
    with open('topic2words.json', 'r') as infile:
        topics2words = json.load(infile)
    topic_count = {dt: 0 for dt in topics2words.keys()}
    for t in topics:
        new_t = words2topics.get(t, None)
        if new_t is not None:
            topic_count[new_t] += 1
    common_topics = {t: c for t, c in topic_count.items() if c > 500}
    print(common_topics)
    # my_dict = {'1': 'aaa', '2': 'bbb', '3': 'ccc'}
    with open('test.csv', 'w') as f:
        for key in common_topics.keys():
            f.write("%s,%s\n"%(key, common_topics[key]))
    with open('topic_count.json', 'w') as outfile:
        json.dump(topic_count, outfile)
    with open('common_topics500.json', 'w') as outfile:
        json.dump(common_topics, outfile)


# *************************
# get data from raw xml (but given the word2topics dictionary

def parse_from_xml(data_path):
    with open(data_path + 'words2topics-new.json', 'r') as infile:
        words2topics = json.load(infile)

    from datetime import time

    data = pd.read_csv(data_path + "Martha_transcripts/index segments for 1000 English Jewish survivor interviews.csv")
    testimonies = set(data['IntCode'])
    topics = [str(d).split("; ") for d in data['IndexedTermLabels']]

    segments = {}
    numtapes = {}
    bad_t = 0
    bad_ts = set()

    # clean testimony list
    for i, t in enumerate(list(testimonies)):
        t_data = data[data['IntCode'] == t]  # data for the specific testimony
        num_tapes = max(t_data['InTapenumber'])
        numtapes[t] = num_tapes  # this is temporary and will be overwritten

        if sum((t_data['InTapenumber'] != t_data['OutTapenumber']).array) > 0:  # a segment goes between tapes
            testimonies.remove(t)
        else:
            times = [time.fromisoformat(tm+'0') for tm in t_data['InTimeCode']]
            time_list = [tm for tm in times if tm.second == 0]  # round times
            if len(time_list) != len(times):
                testimonies.remove(t)

            tape_starts = [tm for tm in times if tm.second == 0 and tm.minute == 0]
            if len(tape_starts) != num_tapes:  # no first segment
                testimonies.remove(t)

    for i, t in enumerate(testimonies):


        t_data = data[data['IntCode'] == t]  # data for the specific testimony
        num_tapes = numtapes[t]
        segments[t] = []  # list of segments for this testimony
        segments_i = 0
        for i in range(1, num_tapes+1):
            segments_i = 0
            t_i_data = t_data[t_data['InTapenumber'] == i]
            try:
                mytree = ET.parse(data_path + f'Martha_transcripts/{t}.{i}.xml')
                last_i = i
            except (FileNotFoundError, ParseError) as err:
                # except:
                if str(err)[:9] != "[Errno 2]":
                    with open(data_path + f'Martha_transcripts/{t}.{i}.xml', 'r', encoding='utf-8') as f:
                        s = f.read().replace("&", " and ")
                    with open(data_path + f'Martha_transcripts/{t}.{i}.xml', 'w', encoding='utf-8') as f:
                        f.write(s)
                    print(err)
                    print(t, i)
                continue
            else:
                myroot = mytree.getroot()
                prev_time = 0
                words = []
                for j, r in enumerate(myroot):
                    for k, e in enumerate(r):
                        if prev_time // 60000 == int(e.attrib['m']) // 60000:
                            if e.text is not None:
                                words.append(e.text)
                        if prev_time // 60000 != int(e.attrib['m']) // 60000 or (j == len(myroot) - 1 and k == len(r) - 1):  # next segment
                            if len(t_data) <= len(segments[t]):  # reached the end
                                terms = "nan"  # !!!  ???
                                bad_ts.add(t)
                            else:
                                terms = str(list(t_data['IndexedTermLabels'])[len(segments[t])])
                                # add some NULLs!!!

                            if terms == "nan" and segments_i == 0:
                                # take NO_TOPIC only for first in a tape
                                terms = ["NO_TOPIC"]
                            else:
                                terms = [words2topics.get(t, None) for t in terms.split('; ') if words2topics.get(t, None) is not None]  # recognized terms
                            bin = str((10 * len(segments[t])) // len(t_data))
                            segments[t].append({'text': ' '.join(words), 'bin': bin, 'terms': terms})
                            segments_i += 1
                            if e.text is not None:
                                words = [e.text]
                            else:
                                words = []
                        prev_time = int(e.attrib['m'])
                while segments_i < len(t_i_data):
                    segments[t].append({'text': "", 'bin': [], 'terms': []})
                    segments_i += 1
        if len(segments[t]) != len(t_data):
            bad_t += 1
            segments.pop(t)
        if t in bad_ts:
            segments.pop(t, None)

    # title_w_segments = [[seg[1][0], ' '.join(seg[0])] for t, s in segments.items() for seg in s if len(seg[1]) == 1]
    segments = {t: [dict for dict in list if len(dict['terms']) == 1] for t, list in segments.items() }
    with open(data_path + 'sf_segments.json', 'w') as outfile:
        json.dump(segments, outfile)

    print("done")


# **************************
# add additional properties

class TestimonyParser:
    def __init__(self, nlp):
        self.referencer, self.srler = Referencer(nlp), SRLer(nlp)
        add_extensions()
        self.nlp = nlp
        logging.info("Made testimony parser")

    def get_pipe(self, name):
        i = self.nlp.pipe_names.index(name)
        return self.nlp.pipeline[i][1]  # the component without the name

    def get_char_spans(self, segments):
        # returns spact spans (for segemnts) given a list of segment texts
        lens = [len(segment)+1 for segment in segments]
        lens[-1] = lens[-1] - 1  # last segment has not extra space
        end_chars = np.cumsum(lens)  # end not included
        start_chars = np.zeros(len(segments), dtype=int)
        start_chars[1:] = end_chars[:-1] + 1  # to skip the extra space
        char_spans = list(zip(start_chars, end_chars))
        return char_spans

    # Define a method that takes a Span as input and returns the Transformer
    # output.
    def span_vector(self, span):
        # Get alignment information for Span. This is achieved by using
        # the 'doc' attribute of Span that refers to the Doc that contains
        # this Span. We then use the 'start' and 'end' attributes of a Span
        # to retrieve the alignment information. Finally, we flatten the
        # resulting array to use it for indexing.
        tensor_ix = span.doc._.trf_data.align[span.start: span.end].data.flatten()
        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = span.doc._.trf_data.tensors[0].shape[-1]
        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
        # Average vectors along axis 0 ("columns"). This yields a 768-dimensional
        # vector for each spaCy Span.
        return tensor.mean(axis=0)

    def make_new_features(self, segment, bin):
        # this is for a new segment (for inference
        # make ent features
        doc = segment.doc
        labels = self.get_pipe("ner").labels
        ent_counts = np.zeros(len(labels))
        for ent in segment.ents:
            ent_counts[labels.index(ent.label_)] += 1
        vec = self.span_vector(segment)
        verbs = self.srler.verbs
        self.srler.add_to_new_span(segment)
        verb_counts, arg0_counts, arg1_counts = np.zeros(len(verbs)), np.zeros(50), np.zeros(50)
        for srl in segment._.srls:
            if srl._.verb and srl._.verb._.verb_id:  # this should always be true
                verb_counts[srl._.verb._.verb_id] += 1
            if srl._.arg0 and srl._.arg0._.arg0_id:
                arg0_counts[srl._.arg0._.arg0_id] += 1
            if srl._.arg1 and srl._.arg1._.arg1_id:
                arg1_counts[srl._.arg1._.arg1_id] += 1

        return list(np.concatenate((ent_counts, verb_counts, arg0_counts, arg1_counts, bin, vec), axis=None))


    def make_features(self, segment, i):
        # make ent features
        doc = segment.doc
        labels = self.get_pipe("ner").labels
        ent_counts = np.zeros(len(labels))
        for ent in segment.ents:
            ent_counts[labels.index(ent.label_)] += 1

        # add span vector and sentiment
        # vec = np.zeros(VECTOR_LEN)
        # if segment.has_vector:
        #     vec[:len(segment.vector[:VECTOR_LEN])] = segment.vector[:VECTOR_LEN]
        #     logging.info(vec)
        # else:
        #     # vec = np.zeros(VECTOR_LEN)
        #     logging.warning("No vector")
        vec = self.span_vector(segment)
        if segment._.srls is None or len(segment._.srls) == 0:  # for a new segment. Separate!!
            verbs = self.srler.verbs
            if doc.spans.get("segments", None) is None:
                bin = 0
            else:
                # bin = int((10 * i) / len(doc.spans["segments"]))
                bin = i  
            verb_counts, arg0_counts, arg1_counts = np.zeros(len(verbs)), np.zeros(50), np.zeros(50)
            return list(np.concatenate((ent_counts, verb_counts, arg0_counts, arg1_counts, bin, vec), axis=None))

        # logging.info(vec)
        # sentiment = segment.sentiment

        # make srl features
        # this is for making the whole testimony from segments
        verbs = self.srler.verbs
        verb_counts, arg0_counts, arg1_counts = np.zeros(len(verbs)), np.zeros(50), np.zeros(50)
        for srl in segment._.srls:
            if srl._.verb and srl._.verb._.verb_id:  # this should always be true
                verb_counts[srl._.verb._.verb_id] += 1
            if srl._.arg0 and srl._.arg0._.arg0_id:
                arg0_counts[srl._.arg0._.arg0_id] += 1
            if srl._.arg1 and srl._.arg1._.arg1_id:
                arg1_counts[srl._.arg1._.arg1_id] += 1

        bin = int((10 * i) / len(doc.spans["segments"]))
        bin_vec = np.zeros(10)
        bin_vec[bin] = 1
        logging.info(f"Lengths (ent, srls, vec, bin): {len(ent_counts)}, {len(verb_counts)+len(arg0_counts)+ len(arg1_counts)}, {len(vec)}, {len(bin_vec)}")
        #  INFO: root:Lengths(ent, srls, ( and len_bin = 1) vec: 18, 3357, 1, 768  # changed!!
        #  INFO: root:Lengths(ent, srls, ( and len_bin = 1) vec: 18, 3357, 768, 10  # new!!
        return list(np.concatenate((ent_counts, verb_counts, arg0_counts, arg1_counts, vec, bin_vec),
                                   axis=None))  # so it will be serializable

    def parse_testimony(self, text):
        doc = self.nlp(text)
        # do coreference resolution
        self.referencer.add_to_Doc(doc, *self.referencer.get_cr(doc.text))
        return doc

    def parse_from_segments(self, texts, labels=None):
        # gets texts for one testimony and returns them as a spacy span with additional attributes
        # add segment list to the doc object. The segments have a pointer to the doc
        logging.info("Making spans...")
        char_spans = self.get_char_spans(texts)
        doc = self.nlp(" ".join(texts))
        # doc.spans['token2segment'] = [doc.char_span(*cs, alignment_mode='expand') for cs in char_spans for _ in range(*cs)]  # a span for each token
        # doc.spans["segments"] = list(set(doc.spans['token2segment']))
        doc.spans["segments"] = [doc.char_span(*cs, alignment_mode='expand') for cs in char_spans]
        logging.info(f"num_segments: {len(doc.spans['segments'])}")
        # also add for each token its segment
        for s in doc.spans["segments"]:
            for t in s:
                t._.segment = s

        # do coreference resolution
        logging.info("Making CR...")
        self.referencer.add_to_Doc(doc, *self.referencer.get_cr(doc.text))
        # do srl
        logging.info("Making srls...")
        for i, s in enumerate(doc.spans["segments"]):
            self.srler.add_to_Span(s, self.srler.parse(s.text))
            s._.feature_vector = self.make_features(s, i)
            if labels:
                s._.real_topic = labels[i]
        logging.info("Made features")

        return doc

    def spacy_parse(self, data_path=None):
        # make the data into spacy span with properties from the whole doc
        with open(data_path + 'sf_segments.json', 'r') as infile:
            data = json.load(infile)

        with open(data_path + 'docs/doc_nums.json', "r") as infile:
            doc_nums = json.load(infile)

        new_data = {}
        # doc_bin = DocBin(store_user_data=True)
        for t, dicts in data.items():
            if t in doc_nums:
                continue
            logging.info(f"Testimony: {t}")
            # texts, bins = list(zip(*[(dict['text'], dict['bin']) for dict in dicts]))  # we will create the bins afterwards
            texts, labels = list(zip(*[(dict['text'], dict['terms']) for dict in dicts]))
            # docs[t] = parse_testimony(nlp, texts)
            # add_extensions()
            doc = self.parse_from_segments(texts, labels=labels)  # we don't transform labels yet
            new_data[t] = self.get_lists(doc)
            doc_nums.append(t)
            # remove_extensions()
            # with open(data_path + "docs/" + doc_names[-1]) as outfile:
            #     pickle.dump(doc, outfile)

            # doc.to_disk(data_path + "docs/" + doc_names[-1])
            with open(data_path + 'docs/doc_nums2.json', "w+") as outfile:
                json.dump(doc_nums, outfile)
            with open(data_path + 'docs/data2.json', "w+") as outfile:  # did I overwrite the old data???
                json.dump(new_data, outfile)

            # doc_bin.add(doc)
            # doc_bin.to_disk(data_path + "docs/data.spacy")
            # self.nlp.vocab.to_disk(data_path + "docs/vocab")

        # docs = list(docs.values())
        # doc_bin = DocBin(docs=docs, store_user_data=True)
        logging.info("Created data")
        return

    def get_lists(self, doc):
        # get a list of segments with the relevant info for saving
        return [(segment.text, segment._.feature_vector, segment._.real_topic) for segment in doc.spans["segments"]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import logging.config
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True, })

    nlp = spacy.load("en_core_web_trf")
    # add_extensions()
    data_path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/'
    # parse_from_xml(data_path)
    parser = TestimonyParser(nlp)
    parser.spacy_parse(data_path)
    # count_topics()

# different format - 19895, 20218, 20367, 20405, 20505, 20873, 20909 etc.