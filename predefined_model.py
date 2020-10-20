import tensorflow as tf
import tensorflow.keras as keras


def conv_block(filters, kernel_size):
    def block_fn(x):
        block = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.04),
            kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')
        )(x)
        block = keras.layers.BatchNormalization(momentum=0.9997, scale=False)(block)
        block = keras.layers.Activation(activation="relu")(block)
        return block

    return block_fn


def dense_block(units):
    def block_fn(x):
        block = keras.layers.Dense(
            units=units,
            kernel_regularizer=keras.regularizers.l2(0.04),
            kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')
        )(x)
        block = keras.layers.BatchNormalization(momentum=0.9997, scale=False)(block)
        block = keras.layers.Activation(activation=keras.activations.relu)(block)

        return block

    return block_fn


def build_vgg(vocab_size):
    keras.backend.clear_session()
    emb_dim = 512
    l_input = keras.layers.Input(shape=(None,))
    imd = keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True)(l_input)

    imd = conv_block(filters=32, kernel_size=3)(imd)
    imd = conv_block(filters=32, kernel_size=3)(imd)
    imd = keras.layers.MaxPool1D()(imd)

    imd = conv_block(filters=64, kernel_size=3)(imd)
    imd = conv_block(filters=64, kernel_size=3)(imd)
    imd = keras.layers.MaxPool1D()(imd)

    imd = conv_block(filters=64, kernel_size=3)(imd)
    imd = conv_block(filters=64, kernel_size=3)(imd)
    imd = keras.layers.MaxPool1D()(imd)

    imd = conv_block(filters=64, kernel_size=3)(imd)   
    imd = conv_block(filters=64, kernel_size=3)(imd)                                                                        
    imd = keras.layers.MaxPool1D()(imd)

    imd = keras.layers.GlobalMaxPooling1D()(imd)

    imd = dense_block(units=128)(imd)
    imd = keras.layers.Dropout(rate=0.5)(imd)
    imd = dense_block(units=128)(imd)
    imd = keras.layers.Dropout(rate=0.5)(imd)

    imd = keras.layers.Dense(units=1)(imd)
    output_tf = keras.layers.Activation(activation=keras.activations.sigmoid)(imd)
    model = keras.models.Model(inputs=l_input, outputs=output_tf)

    return model


def build_inception(vocab_size):
    keras.backend.clear_session()
    emb_dim = 1024

    def inception_block(size=64):
        def block_fn(x):
            branch_0 = conv_block(filters=size, kernel_size=1)(x)

            branch_1 = conv_block(filters=32, kernel_size=1)(x)
            branch_1 = conv_block(filters=64, kernel_size=3)(branch_1)

            branch_2 = conv_block(filters=32, kernel_size=1)(x)
            branch_2 = conv_block(filters=64, kernel_size=5)(branch_2)

            branch_3 = imd
            branch_3 = keras.layers.MaxPool1D(pool_size=2, strides=1, padding="same")(branch_3)
            branch_3 = conv_block(filters=64, kernel_size=1)(branch_3)

            block = keras.layers.Concatenate()([branch_0, branch_1, branch_2, branch_3])
            return block

        return block_fn

    l_input = keras.layers.Input(shape=(None,))
    imd = keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True)(l_input)

    # First Block
    imd = inception_block()(imd)

    # Second Block
    imd = inception_block()(imd)

    # Third Block
    imd = inception_block()(imd)

    # Four Block
    #imd = inception_block()(imd)

    # Flatten
    imd = keras.layers.GlobalMaxPooling1D()(imd)

    #imd = keras.layers.Dense(units=512)(imd)
    #imd = keras.layers.BatchNormalization()(imd)
    #imd = keras.layers.Activation(activation=keras.activations.relu)(imd)
    imd = keras.layers.Dense(units=512)(imd)
    imd = keras.layers.BatchNormalization()(imd)
    imd = keras.layers.Activation(activation=keras.activations.relu)(imd)
    imd = keras.layers.Dropout(rate=0.2)(imd)

    imd = keras.layers.Dense(units=1)(imd)
    output_tf = keras.layers.Activation(activation=keras.activations.sigmoid)(imd)
    model = keras.models.Model(inputs=l_input, outputs=output_tf)

    return model
