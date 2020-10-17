from pipeline import Step
import numpy as np


class Aggregator(Step):

    def __init__(self):
        pass

    def run(self, *args):
        pass


class Merge(Aggregator):

    def __init__(self, need_show_info=True):
        super().__init__()
        self.need_show_info = need_show_info

    def run(self, args: list):
        if len(args) == 0:
            raise ValueError("No data to merge")

        seqs = args[0]
        for i in range(1, len(args)):
            seqs = seqs + args[i]

        seqs = np.array(seqs)

        if self.need_show_info:
            print("Total: {}  example".format(len(seqs)))

