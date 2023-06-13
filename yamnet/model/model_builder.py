from tensorflow.keras import layers, Input, Model, optimizers
from keras_tuner import HyperModel


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
