import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import PreprocessingLayer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import json

from collections.abc import Iterable
import os


fold_idx_path = "data/fold_idx.npy"


class IndexingLayer:
    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab = {}
        self.tokenizer = tokenizer

        return

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.vocab, f)

    @classmethod
    def create_by_file(cls, path, tokenizer=None):
        indexer = IndexingLayer(tokenizer=tokenizer)
        with open(path) as json_file:
            indexer.vocab = json.load(json_file)
        return indexer

    @classmethod
    def create_by_data(cls, data, tokenizer=None):
        indexer = IndexingLayer(tokenizer=tokenizer)
        indexer.build_vocab(data)
        return indexer

    """
    data: should be List of string
    tokenizer: function that convert string to list of token
    """
    def build_vocab(self, data):
        def split_sentence(separator=" "):

            return lambda x: x
            # return lambda x: x.split(separator)

        vocab = set()
        if self.tokenizer is None:
            self.tokenizer = split_sentence(" ")

        for sentence in data:
            words = self.tokenizer(sentence)
            for word in words:
                vocab.add(word)

        self.vocab = {key: index+1 for index, key in enumerate(vocab)}
        return

    def __call__(self, inputs, **kwargs):
        results = []
        for sentence in inputs:
            if type(sentence) == str:
                sentence = self.tokenizer(sentence)

            assert isinstance(sentence, Iterable), "Sequence must be either string or iteration of string"

            results.append([self.vocab[word] if (word in self.vocab) else 0 for word in sentence])

        return np.array(results)


def read_raw_data(path, label):
    seqs = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if ">" not in line:
                seqs.append([line.replace("\n", ""), str(label)])

    return seqs


def get_dataset(path, need_show_info=True):
    # Get neg raw data
    neg_seqs = read_raw_data(os.path.join(path, "negative_data.fasta"), label=0)
    print("# Negative Example: {}".format(len(neg_seqs)))

    # Get pos raw data
    pos_seqs = read_raw_data(os.path.join(path, "positive_data.fasta"), label=1)
    print("# Positive Example: {}".format(len(pos_seqs)))

    train_seqs = np.array(neg_seqs + pos_seqs)

    if need_show_info:
        print("Total: {}  with  {} positive examples and {} negative example"
              .format(len(train_seqs), len(pos_seqs), len(neg_seqs)))

    return train_seqs


def convert_to_ngram(data, n=1):
    ngram_data = []
    for example, label in data:
        ngram_seq = []
        for index in range(len(example)):
            ngram_seq.append(example[index:index + n])

        ngram_seq.append(label)
        ngram_data.append(ngram_seq)

    return np.array(ngram_data)


def load_data():
    DATASET_PATH = "dataset"

    train_raw_path = os.path.join(DATASET_PATH, "raw/train_data")
    test_raw_path = os.path.join(DATASET_PATH, "raw/test_data")

    # Save to csv
    for i in range(1, 5):
        if os.path.isfile(os.path.join(DATASET_PATH, "train_set_{}gram.csv".format(i))):
            continue

        train_set_df = pd.DataFrame(data=convert_to_ngram(get_dataset(train_raw_path), n=i))
        train_set_df.to_csv(os.path.join(DATASET_PATH, "train_set_{}gram.csv".format(i)))
        print()

    for i in range(1, 5):
        if os.path.isfile(os.path.join(DATASET_PATH, "test_set_{}gram.csv".format(i))):
            continue

        test_set_df = pd.DataFrame(data=convert_to_ngram(get_dataset(test_raw_path), n=i))
        test_set_df.to_csv(os.path.join(DATASET_PATH, "test_set_{}gram.csv".format(i)))
        print()

    train_data = pd.read_csv(os.path.join(DATASET_PATH, "train_set_1gram.csv")).values[1:, 1:]
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1].astype(np.float)

    test_data = pd.read_csv(os.path.join(DATASET_PATH, "test_set_1gram.csv")).values[1:, 1:]
    test_x = test_data[:, :-1]
    test_y = test_data[:, -1].astype(np.float)

    return (train_x, train_y), (test_x, test_y)


def get_data():
    (train_x, train_y), (test_x, test_y) = load_data()

    if not os.path.isdir("saved_model"):
        os.mkdir("saved_model")

    if not os.path.isdir("data"):
        os.mkdir("data")

    vocab_path = 'saved_model/vocab.json'
    if os.path.isfile(vocab_path):
        indexer = IndexingLayer.create_by_file(vocab_path)
    else:
        indexer = IndexingLayer.create_by_data(train_x)

        # Save vocab
        indexer.save_vocab(vocab_path)

    train_x_idx = indexer(train_x)
    test_x_idx = indexer(test_x)

    return (train_x_idx, train_y), (test_x_idx, test_y), indexer


def split_k_fold(train_x, train_y, n_fold=5):
    print("START SPLIT DATA TO {} FOLD".format(n_fold))

    folds = StratifiedKFold(
        n_splits=n_fold,
        shuffle=True,
        random_state=0
    ).split(
        train_x,
        train_y
    )

    fold_idx = []
    for fold_index, fold in enumerate(folds):
        fold_idx.append(fold)

    np.save(fold_idx_path, fold_idx)

    return folds


def get_fold_idx(train_x, train_y, n_fold):
    for i in range(n_fold):
        if not os.path.isfile(fold_idx_path):
            split_k_fold(train_x, train_y, n_fold)
            break

    fold_idx = np.load(fold_idx_path, allow_pickle=True)

    return fold_idx


