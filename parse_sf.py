

import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

import pandas as pd


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


def spacy_parse():
    return

def parse2():
    with open('distinct_topics.json', 'r') as infile:
        distinct_topics = json.load(infile)
    topics = {}
    last_topic = None
    word2topic = {}
    topic2words = {}
    with open('new 2.txt', 'r') as infile:
        lines = infile.readlines()
    for i, line in enumerate(lines):
        if line[1:12] == 'margin-left':
            if line[15:17] == '18':
                new_topic = line[18:].strip()
                topic2words[new_topic] = set()
                if new_topic != last_topic:
                    if last_topic is not None:
                        topics[last_topic] = (topics[last_topic], i-1)
                    last_topic = new_topic
                topics[new_topic] = i

            for dt in distinct_topics:
                f = line.find(dt)
                if 20 >= f >= 0 and f + len(dt) + 3 >= len(line):
                # if f >= 0:
                    topic2words[new_topic].add(dt)
                    # word2topic[dt] = new_topic

    topic2words = {t: list(w) for t, w in topic2words.items()}
    words2topic = {w: t for t, words in topic2words.items() for w in words}
    with open('words2topics.json', 'w') as outfile:
        json.dump(words2topic, outfile)
    with open('topic2words.json', 'w') as outfile:
        json.dump(topic2words, outfile)
    print(topics)
    # all_text = ' '.join(lines)
    # for dt in distinct_topics:
    #     all_text.find(dt)

    pass


if __name__ == "__main__":
    data_path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/'
    parse_from_xml(data_path)
    # count_topics()

# different format - 19895, 20218, 20367, 20405, 20505, 20873, 20909 etc.