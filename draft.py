from tensorflow.keras import Model
from tensorflow.keras import layers


def get_draft_model(input_shape):
    # Define input layer
    input_layer = layers.Input(shape=input_shape)

    # Define path 1
    x = layers.LSTM(64, return_sequences=True)(input_layer)
    x = layers.LSTM(64)(x)
    is_tv = layers.Dense(2, activation="softmax")(x)
    output_path1 = layers.Dense(units=input_shape[0] * input_shape[1], activation="softmax")(
        is_tv
    )
    output_path1 = layers.Reshape(target_shape=input_shape)(output_path1)

    # Define path 2
    path2_input = layers.Concatenate()([input_layer, output_path1])
    path2 = layers.LSTM(64, return_sequences=True)(path2_input)
    path2 = layers.LSTM(64)(path2)
    y_pred2 = layers.Dense(2, activation="softmax")(path2)

    # Define model with both paths
    model = Model(inputs=input_layer, outputs=[is_tv, y_pred2])
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    return model
