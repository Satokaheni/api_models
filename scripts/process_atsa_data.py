import pickle

def parse_xml(xml):
    sentences = []
    labels = []
    
    xml = xml.split('<sentence>')
    
    for sample in xml:
        lines = [t.strip() for t in sample.split('\n')]
        text = ''
        polarity = []
        for line in lines:
            if '<text>' in line:
                text = line[6:-7].lower()
            if 'polarity=' in line:
                line = line.split('polarity=')[-1]
                polarity.append(line[1:-3])

        if ('positive' in polarity) and ('negative' in polarity):
            polarity = 'mixed'
        elif 'positive' in polarity:
            polarity = 'positive'
        elif 'negative' in polarity:
            polarity = 'negative'
        else:
            polarity = 'neutral'

        sentences.append(text)
        labels.append(polarity)

    return (sentences, labels)


for file_in, file_out in [('train.xml', 'train.txt'), ('val.xml', 'val.txt'), ('test.xml', 'test.txt')]:
    with open('../data/sentiment/{}'.format(file_in), 'r') as f:
        data = f.read()

    data = parse_xml(data)

    with open('../data/processed/{}'.format(file_out), 'wb') as f:
        pickle.dump(data, f)