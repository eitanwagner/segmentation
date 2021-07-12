
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.tokens import Token

# for CR clusters
# TODO We need pointers to the segments also!!!
Doc.set_extension("clusters", default=None)  # type [Span]  Use Doc.spans instead!!!!
Span.set_extension("ent_type", default=None)  # type String  Use span._label??
Token.set_extension("ent_span", default=None)  # type Span

Span.set_extension("srls", default=None)  # type [Span]
Span.set_extension("arg0", default=None)  # type Span
Span.set_extension("arg1", default=None)  # type Span
Span.set_extension("verb", default=None)  # type Span
Span.set_extension("verb_id", default=None)  # type int (or None if not verb)
Span.set_extension("arg0_id", default=None)  # type int (or None if not an arg0 span)
Span.set_extension("arg1_id", default=None)  # type int (or None if not an arg1 span)

Token.set_extension("srl_span", default=None)  # type Span
Token.set_extension("arg0_span", default=None)  # type Span
Token.set_extension("arg1_span", default=None)  # type Span
Token.set_extension("verb_span", default=None)  # type Span

class Referencer:
    def __init__(self, witness_name="Witness", interviewer_name="Interviewer"):
        from allennlp.predictors.predictor import Predictor
        import allennlp_models.tagging
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
        self.ents = [witness_name, interviewer_name, "Family - mother, father, sister, brother, aunt, uncle", "Nazis, Germans", "Concentration camp, Extermination camp", "Israel, Palastine", "United-states, America"]
        self.nlp = spacy.load("en_core_web_trf")

    def classify_clusters(self, clusters, document, use_len=UnicodeTranslateError):
        # gets list of cluster indices and returns an ent for each cluster
        # document is just text

        # for now just the index by the order!!! (and -1 for over 50)
        if use_len:
            # in this case we return a list of cluster indices by decreasing order, and -1 if no such cluster
            len_ordered = [-1] * 50
            with_lens = [(len(c), i) for i, c in enumerate(clusters)]
            len_ordered[:len(with_lens)] = list(zip(*sorted(with_lens)))[1]
            # len_ordered = list(list(zip(*sorted(with_lens)))[1])
            return len_ordered

        # use number of mentions
        lens = [len(c) for c in clusters]
        largest1 = np.argmax(lens)
        lens.pop(largest1)
        c1 = clusters.pop(largest1)
        largest2 = np.argmax(lens)  # this is after popping, so we need to insert this first
        c2 = clusters.pop(largest2)
        c1, c2 = "Witness", "Interviewer"  # add more validation
        # maybe insted use the name or the 'interviewer' word?

        ent_docs = self.nlp.pipe(self.ents)
        span_clusters = [self.nlp.pipe([" ".join(document[s[0]:s[1]+1]) for s in c]) for c in clusters]  # list of lists
        # for best match
        # max_sims = [self.ents[np.argmax([d.similarity(e) for d in sc for e in ent_docs])] for sc in span_clusters]  # list
        # if only good matches
        max_ents = []
        for sc in span_clusters:
            sims = [d.similarity(e) for d in sc for e in ent_docs]
            if max(sims) > 0.5:
                max_ents.append(self.ents[np.argmax(sims)])
            else:
                max_ents.append("Other")
        max_ents.insert(largest2, c2)
        max_ents.insert(largest1, c1)
        return max_ents

    def get_cr(self, text):
        # this receives the text and not the doc object
        # we assume that the tokens are spacy ones!!!
        cr = self.predictor.predict(text)
        cluster_ents = self.classify_clusters(clusters=cr['clusters'], document=cr['document'])
        return cr['clusters'], cluster_ents

    def add_to_Doc(self, doc, clusters, max_ents):
        # TODO divide into two - add clusters with CR, and then when classifying add the ent_type
        # adds the cluster lists to the doc, and adds the category to each span
        cluster_spans = []
        # doc._.clusters = clusters
        for c, e in zip(clusters, max_ents):
            if e in self.ents:
                for s in c:
                    span = doc[s[0]:s[1]+1]
                    cluster_spans.append(span)
                    span._.ent_type = e
                    for t in span:
                        t._.ent_span = span
        doc._.clusters = cluster_spans  # use doc.span[] instead!!!
        return


class SRLer:
    def __init__(self):
        from allennlp.predictors.predictor import Predictor
        import allennlp_models.structured_prediction.predictors.srl
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        self.nlp = spacy.load("en_core_web_trf")
        self.verbs = (v for v in self.nlp.vocab if v.pos_ == "VERB")

    def parse(self, text, return_locs=True):
        srl = self.predictor.predict(text)
        if return_locs:
            locs_w_tags = [[(i, t) for i, t in enumerate(v['tags']) if t != 'O'] for v in srl['verbs']]
            # these are all of the same length
            arg0_locs = [[i for i, (_, t) in enumerate(l) if t.find("ARG0") != -1] for l in locs_w_tags]  # relative location
            arg1_locs = [[i for i, (_, t) in enumerate(l) if t.find("ARG1") != -1] for l in locs_w_tags]
            v_locs = [[i for i, (_, t) in enumerate(l) if t.find("-V") != -1] for l in locs_w_tags]
            locs = [[i for i, t in l] for l in locs_w_tags]
            return (locs, arg0_locs, arg1_locs, v_locs)

        verb_phrases_w_tags = [[(srl['words'][i], t) for i, t in enumerate(v['tags']) if v != 'O'] for v in srl['verbs']]

        with_arg0 = [(v['description'], i) for i, v in enumerate(srl['verbs']) if v['description'].find("ARG0") != -1]
        arg0s = [(d.split("[B-ARG0: ", 1)[1].split("]")[0], i) for d, i in with_arg0]  # a list of all ARG0 phrases in tuple with verb index
        with_arg1 = [(v['description'], i) for i, v in enumerate(srl['verbs']) if v['description'].find("ARG1") != -1]
        arg1s = [(d.split("[B-ARG1: ", 1)[1].split("]")[0], i) for d, i in with_arg1]
        return verb_phrases_w_tags, arg0s, arg1s

    def parse_simple(self, text):
        # returns a list of verb phrases for this text. Does not consider the roles
        srl = self.predictor.predict(text)
        verb_phrases = [" ".join([srl['words'][i] for i, t in enumerate(v['tags']) if v != 'O']) for v in srl['verbs']]
        return verb_phrases

    def add_to_Span(self, span, loc_tuples):
        locs, arg0_locs, arg1_locs, v_locs = loc_tuples
        # gets spacy span (segment) and adds the srl attribute to each span
        srls = []
        # doc._.clusters = clusters
        for l, a0, a1, v in zip(locs, arg0_locs, arg1_locs, v_locs):
            srl_span = span[l[0]:l[-1]]
            srls.append(srl_span)
            for t in srl_span:
                t._.srl_span = srl_span
            if len(a0) > 0:
                arg0_span = srl_span[a0[0]:a0[-1]+1]
                srl_span._.arg0 = arg0_span
                arg0_span._.arg0_id = arg0_span._.ent_type
                for t in arg0_span:
                    t._.arg0_span = arg0_span
            if len(a1) > 0:
                arg1_span = srl_span[a1[0]:a1[-1]+1]
                srl_span._.arg1 = arg1_span
                arg1_span._.arg1_id = arg1_span._.ent_type
                for t in arg1_span:
                    t._.arg1_span = arg1_span
            if len(v) > 0:
                verb_span = srl_span[v[0]:v[-1]+1]
                srl_span._.verb = verb_span
                for t in verb_span:
                    t._.verb_span = verb_span
                    if t.pos_ == "VERB" and t.lemma_ in self.verbs:  # if more than one with pos verb then takes the last!!
                        srl_span._.verb_id = self.verbs.index(t.lemma_)
        span._.srls = srls
        return