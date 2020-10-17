import tensorflow.keras as keras
import tensorflow as tf
from models.metrics import BinaryF1Score, BinaryMCC, BinaryAccuracy, BinarySensitivity, BinarySpecificity


batch_size = 32
n_epoch = 40
class_weight = {0: 1, 1: 3}
lr = 0.01


class Model(object):
    def __init__(self, model: keras.models.Model, log_dir):
        self.callbacks = {}
        self.model = model
        self.build_callbacks(log_dir)

    def compile(self):
        self.model.compile(
            loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(lr),
            metrics=[
                BinaryAccuracy(),
                BinarySpecificity(),
                BinarySensitivity(),
                BinaryMCC(),
            ]
        )

    def build_callbacks(self, log_dir):
        self.callbacks["early_stop"] = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-5,
            patience=5,
            verbose=2,
            mode='auto',
            baseline=None,
            restore_best_weights=False
        )

        self.callbacks["tensorboard"] = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=2,
            embeddings_freq=0,
            embeddings_metadata=None,
        )

        def scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
                # return lr * tf.math.exp(-0.01)

        self.callbacks["lr_decay"] = tf.keras.callbacks.LearningRateScheduler(scheduler)

    def fit(self, train_x, train_y):
        self.compile()

        self.model.fit(
            train_x,
            train_y,
            class_weight=class_weight,
            batch_size=batch_size,
            epochs=n_epoch,
            callbacks=[
                self.callbacks["tensorboard"],
                self.callbacks["lr_decay"]
            ],
            shuffle=True,
            verbose=1
        )

    def evaluate(self, test_x, test_y):
        return self.model.evaluate(test_x, test_y, verbose=1)

    def predict(self, x):
        pass

    def save(self, model_path):
        self.model.save(model_path)



