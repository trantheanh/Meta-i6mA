from pipeline.load_data import XLSXLoader, CSVLoader
from pipeline.filter import Filter, RGDatasetFilter
from pipeline.transform import Transformer, CSVRaw, BinaryLabels, CharacterFeatures, Indexer
from pipeline.evaluate import Evaluator
from pipeline.split_data import Splitter
from pipeline.save_data import NgramsSaver
from pipeline.aggregate_data import Aggregator
from pipeline import Step
from models.model import Model
from predefined_model import build_vgg, build_inception
import os
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def read_pipeline(pipeline):
    result = None
    for step in pipeline:
        if issubclass(type(step), Step):
            result = step.run(result)

        if issubclass(type(step), list):
            result = [substep.run(result) for substep in step]

    return result


def run():
    train_pipeline = [
        XLSXLoader("data/Meta-6mA-datasets.xlsx", n_sheet=4, header=None),
        NgramsSaver(
            paths=[
                "data/RG-Training-datasets.csv",
                "data/RG-Independent-datasets.csv",
                "data/Rice-datasets.csv",
                "data/Arabidopsis-datasets.csv"
            ]
        ),
        CSVLoader(
            path="data/RG-Training-datasets.csv",
            ngram=1
        ),
        RGDatasetFilter(),
        CSVRaw(),
        CharacterFeatures(),
        BinaryLabels("6mA"),
        Indexer(path="saved_model/RG/vocab.json")
        # Filter(),
        # Transformer(),
        # Splitter(),
        # Aggregator(),
        # Evaluator()
    ]

    train_x, train_y = read_pipeline(train_pipeline)

    model_fn = build_vgg

    model = Model(
        model=model_fn((len(Indexer(path="saved_model/RG/vocab.json").instance.vocab) + 1)),
        log_dir="log/"
    )

    test_pipeline = [
        CSVLoader(
            path="data/RG-Independent-datasets.csv",
            ngram=1
        ),
        RGDatasetFilter(),
        CSVRaw(),
        CharacterFeatures(),
        BinaryLabels("6mA"),
        Indexer(path="saved_model/RG/vocab.json")
    ]

    model.test_set = read_pipeline(test_pipeline)
    model.fit(
        train_x,
        train_y,
    )

    result = model.evaluate(model.test_set[0], model.test_set[1])
    print(result)


run()





