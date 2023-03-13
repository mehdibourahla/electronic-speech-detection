import keras
from keras import layers


def get_model(input_shape):
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=input_shape)
    # Embed each integer in a 128-dimensional vector
    # Add 2 bidirectional LSTMs
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    # Add a classifier
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    return model
