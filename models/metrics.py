import tensorflow.keras as keras
import tensorflow as tf
import sklearn.metrics as metrics


"""# Build Metrics"""


class ConfusionMatrix(keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='confusion_matrix', **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="true_positives", initializer='zeros')
        self.tn = self.add_weight(name="true_negatives", initializer='zeros')
        self.fp = self.add_weight(name="false_positives", initializer='zeros')
        self.fn = self.add_weight(name="false_negatives", initializer='zeros')
        self.threshold = threshold

    def update_state(self, y_true, y_pred, **kwargs):
        y_pred = tf.cast(y_pred > tf.cast(self.threshold, y_pred.dtype), y_pred.dtype)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.cast(tp, self.dtype)
        self.tp.assign_add(tf.reduce_sum(tp))

        tn = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        tn = tf.cast(tn, self.dtype)
        self.tn.assign_add(tf.reduce_sum(tn))

        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fp = tf.cast(fp, self.dtype)
        self.fp.assign_add(tf.reduce_sum(fp))

        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        fn = tf.cast(fn, self.dtype)
        self.fn.assign_add(tf.reduce_sum(fn))

    def result(self):
        return self.tp, self.tn, self.fp, self.fn


class BinaryAccuracy(ConfusionMatrix):
    def __init__(self, threshold=0.5, name='acc', **kwargs):
        super(BinaryAccuracy, self).__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        numerator = self.tp + self.tn
        denominator = self.tp + self.fp + self.tn + self.fn

        return tf.divide(numerator, denominator)


class BinaryMCC(ConfusionMatrix):
    def __init__(self, threshold=0.5, name='mcc', **kwargs):
        super(BinaryMCC, self).__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        numerator = self.tp * self.tn - self.fp * self.fn
        denominator = tf.sqrt(
            (self.tp+self.fp) * (self.tp+self.fn) * (self.tn+self.fp) * (self.tn+self.fn)
        ) + tf.keras.backend.epsilon()

        return tf.divide(numerator, denominator)


class BinarySensitivity(ConfusionMatrix):
    def __init__(self, threshold=0.5, name='sen', **kwargs):
        super(BinarySensitivity, self).__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        numerator = self.tp
        denominator = self.tp + self.fn + tf.keras.backend.epsilon()

        return tf.divide(numerator, denominator)


class BinarySpecificity(ConfusionMatrix):
    def __init__(self, threshold=0.5, name='spec', **kwargs):
        super(BinarySpecificity, self).__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        numerator = self.tn
        denominator = self.tn + self.fp + tf.keras.backend.epsilon()

        return tf.divide(numerator, denominator)


class BinaryF1Score(ConfusionMatrix):
    def __init__(self, threshold=0.5, name='f1', **kwargs):
        super(BinaryF1Score, self).__init__(threshold=threshold, name=name, **kwargs)

    def result(self):
        precision = tf.divide(self.tp, self.tp + self.fp)
        recall = tf.divide(self.tp, self.tp + self.fn)

        f1 = 2 * tf.divide(precision * recall, precision + recall)
        return f1