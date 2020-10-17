from pipeline import Step
import pandas as pd
import numpy as np


class Loader(Step):

    def __init__(self):
        pass

    def run(self, *args):
        pass


class FastaLoader(Loader):

    def __init__(self, path, label):
        super(FastaLoader, self).__init__()
        self.path = path
        self.label = label

    def run(self, *args):
        seqs = []
        with open(self.path, "r") as f:
            for i, line in enumerate(f):
                if ">" not in line:
                    seqs.append([line.replace("\n", ""), str(self.label)])

        return seqs


class CSVLoader(Loader):

    def __init__(self, path, ngram=1, header=None):
        super(CSVLoader, self).__init__()
        self.path = path
        self.ngram = ngram
        self.header = header

    def run(self, *args):
        train_data = pd.read_csv(self.path.format(self.ngram)).values

        return train_data


class XLSXLoader(Loader):
    def __init__(self, path, n_sheet, header=None):
        super(XLSXLoader, self).__init__()
        self.path = path
        self.n_sheet = n_sheet
        self.header = header

    def run(self, *args):
        data = []
        for i in range(self.n_sheet):
            data.append(pd.read_excel(self.path, header=self.header, sheet_name=i))

        return data

    


