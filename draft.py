from tensorflow.keras import Model
from tensorflow.keras import layers


def get_model(input_shape):
    # Define input layer
    input_layer = layers.Input(shape=input_shape)

    # Define the shared feature extraction layer
    shared_layer = layers.Dense(128, activation="relu")(input_layer)

    # Define path 1
    x = layers.LSTM(64, return_sequences=True)(shared_layer)
    x = layers.LSTM(64)(x)
    output_layer_1 = layers.Dense(2, activation="softmax", name="task_1")(x)

    output_path1 = layers.Dense(
        units=input_shape[0] * input_shape[1], activation="softmax"
    )(output_layer_1)
    output_path1 = layers.Reshape(target_shape=input_shape)(output_path1)

    # Define path 2
    path2_input = layers.Concatenate()([shared_layer, output_path1])
    path2 = layers.LSTM(64, return_sequences=True)(path2_input)
    path2 = layers.LSTM(64)(path2)
    output_layer_2 = layers.Dense(2, activation="softmax", name="task_2")(path2)

    # Define model with both paths
    model = Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])
    model.compile(
        "adam",
        loss={
            "task_1": "categorical_crossentropy",
            "task_2": "categorical_crossentropy",
        },
        metrics=["accuracy"],
    )

    return model
