from pipeline import Step


class Splitter(Step):

    def __init__(self):
        print("INIT")

    def run(self, arg):
        print("RUN")

