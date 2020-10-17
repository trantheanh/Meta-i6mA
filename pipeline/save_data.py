from pipeline import Step
import pandas as pd
import os


class Saver(Step):

    def __init__(self):
        pass

    def run(self, *args):
        pass


class NgramSaver(Saver):
    def __init__(self, path, ngram=1):
        super(NgramSaver, self).__init__()
        self.path = path
        self.ngram = ngram

    def run(self, data):
        train_set_df = pd.DataFrame(data=data)
        train_set_df.to_csv(self.path.format(self.ngram))


class NgramsSaver(Saver):
    def __init__(self, paths, ngram=1):
        super(NgramsSaver, self).__init__()
        self.paths = paths
        self.ngram = ngram

    def run(self, data):
        for i in range(len(self.paths)):
            train_set_df = pd.DataFrame(data=data[i])
            train_set_df.to_csv(self.paths[i].format(self.ngram))
