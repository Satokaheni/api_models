import pickle
from collections import Counter
import sys

label_map = {'positive': 3, 'negative': 2, 'neutral': 1}

def parse_xml(xml):
    sentences = []
    labels = []
    
    xml = xml.split('<sentence>')
    
    for sample in xml[1:]:
        lines = [t.strip() for t in sample.split('\n')]
        text = ''
        label = []
        for line in lines:
            if '<text>' in line:
                text = line[6:-7].lower().strip()
                label = ([0] * len(text))
            if 'polarity=' in line:
                line = line.split()
                start = int(line[1].split('from="')[1].split('"')[0])
                end = int(line[-1].split('to="')[1].split('"')[0])
                polarity = line[2].split('polarity="')[-1][:-1]
                for j in range(start, end+1):
                    label[j] = label_map[polarity]

        sentences.append(text)
        ner_label = []
        start = 0
        for i in range(0, len(text)):
            if text[i] == ' ':
                most_common, _ = Counter(label[start:i]).most_common(1)[0]
                ner_label.append(most_common)
                start = i

        most_common, _ = Counter(label[start:]).most_common(1)[0]
        ner_label.append(most_common)
                
        labels.append(ner_label)
        
    return (sentences, labels)


for file_in, file_out in [('train.xml', 'train.txt'), ('val.xml', 'val.txt'), ('test.xml', 'test.txt')]:
    with open('../data/targeted/{}'.format(file_in), 'r') as f:
        data = f.read()

    data = parse_xml(data)

    with open('../data/targeted/{}'.format(file_out), 'wb') as f:
        pickle.dump(data, f)