import tensorflow as tf
import tensorflow.keras as keras

from preprocessing import get_data, get_fold_idx
from models.metrics import BinaryF1Score, BinaryMCC, BinaryAccuracy, BinarySensitivity, BinarySpecificity
from matplotlib import pyplot as plt
import sklearn.metrics as metrics

import numpy as np
import pandas as pd


# n_fold = 5
# (train_x_idx, train_y), (test_x_idx, test_y), indexer = get_data()
# folds = get_fold_idx(train_x_idx, train_y, n_fold=n_fold)
#
# train_sen_idx = 3
# train_spec_idx = 2
# test_sen_idx = 7
# test_spec_idx = 6


def load_model(
        model_path,
):
    model = keras.models.load_model(
        model_path,
        compile=False
    )
    return model


def compile_by(model, threshold):
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=[
            BinaryAccuracy(threshold),
            BinarySpecificity(threshold),
            BinarySensitivity(threshold),
            BinaryMCC(threshold),
        ]
    )

    return model


def evaluate(fold_index=-1, smooth=300):
    if fold_index >= len(folds):
        return

    if fold_index == -1:
        fold_train_x = train_x_idx
        fold_train_y = train_y

        fold_dev_x = test_x_idx
        fold_dev_y = test_y

        model_path = "saved_model/model_FINAL.h5"
    else:
        train_idx, dev_idx = folds[fold_index]

        fold_train_x = train_x_idx[train_idx]
        fold_train_y = train_y[train_idx]

        fold_dev_x = train_x_idx[dev_idx]
        fold_dev_y = train_y[dev_idx]

        model_path = "saved_model/model_fold_{}.h5".format(fold_index)

    model: keras.models.Model = load_model(model_path)
    fold_train_pred = model.predict(fold_train_x, verbose=1).flatten()
    fold_dev_pred = model.predict(fold_dev_x, verbose=1).flatten()

    results = []
    for i in range(smooth):
        threshold = i * 1/smooth
        acc = BinaryAccuracy(threshold)
        spec = BinarySpecificity(threshold)
        sen = BinarySensitivity(threshold)
        mcc = BinaryMCC(threshold)

        acc.reset_states()
        spec.reset_states()
        sen.reset_states()
        mcc.reset_states()

        acc.update_state(fold_train_y, fold_train_pred)
        spec.update_state(fold_train_y, fold_train_pred)
        sen.update_state(fold_train_y, fold_train_pred)
        mcc.update_state(fold_train_y, fold_train_pred)

        result = [threshold, acc.result().numpy(), spec.result().numpy(), sen.result().numpy(), mcc.result().numpy()]

        acc.reset_states()
        spec.reset_states()
        sen.reset_states()
        mcc.reset_states()

        acc.update_state(fold_dev_y, fold_dev_pred)
        spec.update_state(fold_dev_y, fold_dev_pred)
        sen.update_state(fold_dev_y, fold_dev_pred)
        mcc.update_state(fold_dev_y, fold_dev_pred)

        result = result + [acc.result().numpy(), spec.result().numpy(), sen.result().numpy(), mcc.result().numpy()]

        results.append(result)

    return results


def evaluate_on_kfold():
    for fold_index in range(n_fold):
        np.savetxt("result/auc_data_fold_{}.csv".format(fold_index), np.array(evaluate(fold_index, smooth=300)), delimiter=",")

    np.savetxt("result/auc_data.csv", np.array(evaluate(-1, smooth=300)), delimiter=",")


def draw_auc(fold_index, n_fold=5):
    if fold_index == -1:
        results = pd.read_csv("result/auc_data.csv", header=None, delimiter=" ").values
        train_label = "train_set"
        test_label = "test_set"
    else:
        results = pd.read_csv(
            "result/auc_data_fold_{}.csv".format(0),
            header=None,
            delimiter=" "
        ).values/n_fold
        for i in range(1, n_fold):
            result = pd.read_csv(
                "result/auc_data_fold_{}.csv".format(fold_index),
                header=None,
                delimiter=" "
            ).values
            results = results + result/n_fold

        train_label = "train_set"
        test_label = "cv_set"

    train_FPR = 1 - results[:, train_spec_idx]
    train_TPR = results[:, train_sen_idx]
    train_auc = metrics.auc(train_FPR, train_TPR)

    test_FPR = 1 - results[:, test_spec_idx]
    test_TPR = results[:, test_sen_idx]
    test_auc = metrics.auc(test_FPR, test_TPR)

    print("AUC: {} {}".format(train_auc, test_auc))

    _, ax = plt.subplots()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.title('Receiver Operating Characteristic')

    train_plot, = plt.plot(train_FPR, train_TPR, "b", label=train_label + " AUC = {0:.5f}".format(train_auc))
    test_plot, = plt.plot(test_FPR, test_TPR, "r", label=test_label + " AUC = {0:.5f}".format(test_auc))

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(handles=[train_plot, test_plot], loc='lower right')
    plt.show()


# draw_auc(0)
# draw_auc(-1)
# model_path = "saved_model/model_FINAL.h5"
# models: keras.models.Model = load_model(model_path)
# models.compile(
#         loss=keras.losses.binary_crossentropy,
#         optimizer=keras.optimizers.Adam(),
#         metrics=[
#             BinaryAccuracy(),
#             BinarySpecificity(),
#             BinarySensitivity(),
#             BinaryMCC(),
#             keras.metrics.AUC(num_thresholds=300)
#         ]
#     )
# models.evaluate(train_x_idx, train_y)
# models.evaluate(test_x_idx, test_y)

from pipeline import Step


class Evaluator(Step):

    def __init__(self):
        print("INIT")

    def run(self, arg):
        print("RUN")

