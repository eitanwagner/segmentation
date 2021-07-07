

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

def parse1():
    with open('words2topics.json', 'r') as infile:
        words2topics = json.load(infile)
    with open('common_topics500.json', 'r') as infile:
        new_topics = list(json.load(infile).keys())
    from datetime import time

    topic2num = pd.read_csv('topic2num.csv', header=None, index_col=0, squeeze=True).to_dict()
    num2newtopic = pd.read_csv('num2newtopic.csv', header=None, index_col=0, squeeze=True).to_dict()
    new_words2topics = {w: num2newtopic.get(topic2num.get(t, None), None) for w, t in words2topics.items()}

    data = pd.read_csv("Martha_transcripts/index segments for 1000 English Jewish survivor interviews.csv")
    # print(data.head())

    testimonies = set(data['IntCode'])
    topics = [str(d).split("; ") for d in data['IndexedTermLabels']]
    print(len(topics))

    distinct_topics = set([t for topic in topics for t in topic])
    print(len(distinct_topics))
    with open('distinct_topics.json', 'w+') as f:
        json.dump(list(distinct_topics), f)

    segments = {}
    wrong_len = []
    bad_t = 0
    not_round = 0
    last_i = 0
    # time_list = []
    for _, t in enumerate(testimonies):
        t_data = data[data['IntCode'] == t]
        num_tapes = max(t_data['InTapenumber'])
        if sum((t_data['InTapenumber'] != t_data['OutTapenumber']).array) > 0:
            wrong_len.append(t)
            not_round += 1
            continue
        else:
            times = [time.fromisoformat(tm+'0') for tm in t_data['InTimeCode']]
            time_list = [tm for tm in times if tm.second == 0]
            if len(time_list) != len(times):
                # wrong_len.append(t)
                not_round += 1
                continue
            tape_starts = [tm for tm in times if tm.second == 0 and tm.minute == 0]
            if len(tape_starts) != num_tapes:
                # wrong_len.append(t)
                continue


        segments[t] = []
        for i in range(1, 20):
            segments_i = 0
            t_i_data = t_data[t_data['InTapenumber'] == i]
            try:
                # mytree = ET.parse(f'Martha_transcripts/{t}.{i}.xml')
                mytree = ET.parse(f'Martha_transcripts/{t}.{i}.xml')
                last_i = i
            except (FileNotFoundError, ParseError) as err:
                # except:
                if str(err)[:9] != "[Errno 2]":
                    with open(f'Martha_transcripts/{t}.{i}.xml', 'r', encoding='utf-8') as f:
                        s = f.read().replace("&", " and ")
                    with open(f'Martha_transcripts/{t}.{i}.xml', 'w', encoding='utf-8') as f:
                        f.write(s)
                    print(err)
                    print(t, i)
                continue
            else:
                myroot = mytree.getroot()
                prev_t = 0
                words = []
                for r in myroot:
                    for e in r:
                        if prev_t // 60000 == int(e.attrib['m']) // 60000:
                            if e.text is not None:
                                words.append(e.text)
                        else:
                            if len(t_data) <= len(segments[t]):
                                terms = "nan"
                            else:
                                terms = str(list(t_data['IndexedTermLabels'])[len(segments[t])])
                            # if math.isnan(terms):
                            #     terms = None
                            # else:
                            # terms = [words2topics.get(t, None) for t in terms.split('; ') if words2topics.get(t, None) in new_topics]
                            terms = [new_words2topics.get(t, None) for t in terms.split('; ') if new_words2topics.get(t, None) is not None]
                            bin_token = "BIN_" + str((10 * len(segments[t])) // len(t_data))
                            segments[t].append([words + [bin_token], terms])
                            segments_i += 1
                            if e.text is not None:
                                words = [e.text]
                            else:
                                words =[]
                        prev_t = int(e.attrib['m'])
                if len(words) > 0:
                    if len(t_data) <= len(segments[t]):
                        terms = "nan"
                    else:
                        terms = str(list(t_data['IndexedTermLabels'])[len(segments[t])])
                    # terms = [words2topics.get(t, None) for t in terms.split('; ') if words2topics.get(t, None) is not None]
                    terms = [new_words2topics.get(t, None) for t in terms.split('; ') if new_words2topics.get(t, None) is not None]
                    segments[t].append([words, terms])
                    # segments[t].append(list(words))
                    segments_i += 1
                    words = []
                while segments_i < len(t_i_data):
                    segments[t].append([[], []])
                    segments_i += 1
        if len(segments[t]) != len(t_data):
            bad_t += 1
            segments.pop(t)

    title_w_segments = [[seg[1][0], ' '.join(seg[0])] for t, s in segments.items() for seg in s if len(seg[1]) == 1]
    with open('sf_title_w_segments1.json', 'w') as outfile:
        json.dump(title_w_segments, outfile)

    # print(len(wrong_len))
    # print("All segments: ", sum([len(segment) for segment in segments.values()]))
    print("done")


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
    parse1()
    # count_topics()

# different format - 19895, 20218, 20367, 20405, 20505, 20873, 20909 etc.