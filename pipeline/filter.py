from pipeline import Step


class Filter(Step):

    def __init__(self):
        pass

    def run(self, *args):
        pass


class RGDatasetFilter(Filter):
    def __init__(self):
        super(RGDatasetFilter, self).__init__()

    def run(self, data):
        result = data[3:, 1:]
        return result


