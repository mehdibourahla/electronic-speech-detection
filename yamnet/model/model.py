import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, Input, Model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

os.environ["TFHUB_CACHE_DIR"] = "/users/mbourahl/.cache"


np.random.seed(42)
tf.random.set_seed(42)


# LSTM Model
def lstm_model(input_shape):
    adam = optimizers.Adam(3e-4)
    inputs = Input(shape=input_shape)

    # Using glorot_uniform (Xavier) for LSTM and he_normal for Dense with softmax
    x = layers.LSTM(
        64,
        return_sequences=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
    )(inputs)
    x = layers.LSTM(
        64, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal"
    )(x)
    outputs = layers.Dense(
        2, activation="softmax", kernel_initializer="glorot_uniform"
    )(x)

    model = Model(inputs, outputs)
    model.compile(adam, "categorical_crossentropy", metrics=["accuracy"])

    return model


def evaluate_model(model, test_generator):
    y_pred = []
    y_test = []
    for i in range(len(test_generator)):
        x, y = test_generator[i]
        y_pred.extend(model.predict(x))
        y_test.extend(y)

    # Convert lists to arrays
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)

    # Convert predictions from one-hot to labels
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }

    return results


def get_early_stopping_callback(monitor="val_loss", patience=10):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=0,
        patience=patience,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )
    return early_stopping_callback


def plot_history(history, output_dir):
    output_path = f"{output_dir}/lstm_history.png"
    plt.figure(figsize=(12, 8))

    # Plot the training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("LSTM Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper right")

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("LSTM Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{output_path}")


def train_model(
    args,
    training_generator,
    validation_generator=None,
):
    # Callbacks
    early_stopping_callback = get_early_stopping_callback(monitor="val_loss")

    # Get a batch of data
    data_batch, _ = training_generator.__getitem__(0)

    # Get the shape of a single sample
    input_shape = data_batch[0].shape
    model = lstm_model(input_shape)

    history = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=args.epochs,
        callbacks=[early_stopping_callback],
    )

    return history, model
