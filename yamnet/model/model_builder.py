from tensorflow.keras import layers, Input, Model, optimizers
from keras_tuner import HyperModel
from transformer import TransformerClassifier


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        inputs = Input(shape=self.input_shape)
        x = layers.Conv1D(
            filters=hp.Int("conv_1_filter", min_value=64, max_value=128, step=32),
            kernel_size=hp.Choice("conv_1_kernel", values=[2, 3]),
            activation="relu",
        )(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(2, activation="softmax")(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer=optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-5,
                    max_value=1e-3,
                    sampling="LOG",
                    default=3e-4,
                )
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model


class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape

    def build(self, hp):
        inputs = Input(shape=self.input_shape)

        x = layers.LSTM(
            units=hp.Int("units_1", min_value=32, max_value=128, step=32),
            return_sequences=True,
            input_shape=self.input_shape,
        )(inputs)
        x = layers.LSTM(units=hp.Int("units_2", min_value=32, max_value=128, step=32))(
            x
        )
        outputs = layers.Dense(2, activation="softmax")(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-5,
                    max_value=1e-3,
                    sampling="LOG",
                    default=3e-4,
                )
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model


class TransformerHyperModel(HyperModel):
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes

    def build(self, hp):
        num_layers = hp.Int("num_layers", min_value=1, max_value=4, step=1)
        embed_dim = self.input_dim[1]
        num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)
        ff_dim = hp.Int("ff_dim", min_value=256, max_value=1024, step=256)
        input_shape = (None, embed_dim)

        model = TransformerClassifier(
            num_layers, embed_dim, num_heads, ff_dim, input_shape, self.num_classes
        )

        model.compile(
            optimizer=optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-5,
                    max_value=1e-3,
                    sampling="LOG",
                    default=3e-4,
                )
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
