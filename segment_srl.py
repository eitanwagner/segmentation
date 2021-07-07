
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.tokens import Token

# for CR clusters
Doc.set_extension("clusters", default=None)  # type [Span]  Use Doc.spans instead!!!!
Span.set_extension("ent_type", default=None)  # type String  Use span._label??
Token.set_extension("ent_span", default=None)  # type Span

Span.set_extension("srls", default=None)  # type [Span]
Span.set_extension("arg0", default=None)  # type Span
Span.set_extension("arg1", default=None)  # type Span
Span.set_extension("verb", default=None)  # type Span
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
        import spacy
        self.nlp = spacy.load("en_core_web_md")

    def classify_clusters(self, clusters, document):
        # gets list of cluster indices and returns an ent for each cluster
        # document is just text

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

    def get_cr(self, doc):
        cr = self.predictor.predict(doc)
        cluster_ents = self.classify_clusters(clusters=cr['clusters'], document=cr['document'])

    def add_to_Doc(self, doc, clusters, max_ents):
        # TODO divide into two - add clusters with CR, and then when classifying add the ent_type
        # adds the cluster lists to the doc, and adds the category to each span
        cluster_spans = []
        # doc._.clusters = clusters
        for c, e in zip(clusters, max_ents):
            if e in self.ents:
                for s in c:
                    cluster_spans.append(doc[s[0]:s[1]+1])
                    doc[s[0]:s[1]+1]._.ent_type = e
                    for t in doc[s[0]:s[1]+1]:
                        t._.ent_span = doc[s[0]:s[1]+1]
        doc._.clusters = cluster_spans
        return


class SRLer:
    def __init__(self):
        from allennlp.predictors.predictor import Predictor
        import allennlp_models.structured_prediction.predictors.srl
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

    def parse(self, text, return_loc=False):
        srl = self.predictor.predict(text)
        if return_loc:
            locs_w_tags = [[(i, t) for i, t in enumerate(v['tags']) if t != 'O'] for v in srl['verbs']]
            # put these with the srls!!!
            arg0_locs = [[i for i, (_, t) in enumerate(l) if t.find("ARG0") != -1] for l in locs_w_tags]  # relative location
            arg1_locs = [[i for i, (_, t) in enumerate(l) if t.find("ARG1") != -1] for l in locs_w_tags]
            v_locs = [[i for i, (_, t) in enumerate(l) if t.find("-V") != -1] for l in locs_w_tags]
            locs = [[i for i, t in l] for l in locs_w_tags]
            return locs, arg0_locs, arg1_locs, v_locs

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

    def add_to_Span(self, span, locs, arg0_locs, arg1_locs, v_locs):
        # gets spacy span (segment) and adds the srl attribute to each span
        srls = []
        # doc._.clusters = clusters
        for l, a0, a1, v in zip(locs, arg0_locs, arg1_locs, v_locs):
            srl_span = span[l[0]:l[-1]]
            srls.append(srl_span)
            for t in srl_span:
                t._.srl_span = srl_span
            if len(a0) > 0:
                srl_span._.arg0 = srl_span[a0[0]:a0[-1]+1]
                for t in srl_span[a0[0]:a0[-1]+1]:
                    t._.arg0_span = srl_span[a0[0]:a0[-1]+1]
            if len(a1) > 0:
                srl_span._.arg1 = srl_span[a1[0]:a1[-1]+1]
                for t in srl_span[a1[0]:a1[-1]+1]:
                    t._.arg1_span = srl_span[a1[0]:a1[-1]+1]
            if len(v) > 0:
                srl_span._.verb = srl_span[v[0]:v[-1]+1]
                for t in srl_span[v[0]:v[-1]+1]:
                    t._.verb_span = srl_span[v[0]:v[-1]+1]
        span._.srls = srls
        return