from pipeline import Step
import numpy as np
import os
from preprocessing import IndexingLayer


class Transformer(Step):

    def __init__(self):
        pass

    def run(self, *arg):
        pass


class Ngram(Transformer):

    def __init__(self, ngram=1):
        super().__init__()
        self.ngram = ngram

    def run(self, data):
        ngram_data = []
        for example, label in data:
            ngram_seq = []
            for index in range(len(example)):
                ngram_seq.append(example[index:index + self.ngram])

            ngram_seq.append(label)
            ngram_data.append(ngram_seq)

        return np.array(ngram_data)


class CSVRaw(Transformer):
    def __init__(self):
        super(CSVRaw, self).__init__()

    def run(self, data):
        x = data[:, :-1]
        y = data[:, -1]
        return x, y


class CharacterFeatures(Transformer):
    def __init__(self):
        super(CharacterFeatures, self).__init__()

    def run(self, data):
        X, Y = data

        return [list(x[0]) for x in X], Y


class BinaryLabels(Transformer):
    def __init__(self, pos_label):
        super(BinaryLabels, self).__init__()
        self.pos_label = pos_label

    def run(self, data):
        X, Y = data

        return X, (Y == self.pos_label).astype(float)


class Indexer(Transformer):
    __cached = {}

    def __init__(self, path):
        super(Indexer, self).__init__()
        self.instance = None

        if os.path.isfile(path):
            self.instance = IndexingLayer.create_by_file(path)

        self.path = path

    def run(self, data):
        x, y = data

        if not os.path.isfile(self.path):
            os.mkdir(os.path.dirname(self.path))
            self.instance = IndexingLayer.create_by_data(x)

            # Save vocab
            self.instance.save_vocab(self.path)

        return self.instance(x), y







