import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from transformer import TransformerClassifier
from tensorflow.keras import layers, optimizers, Input, Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator

os.environ["TFHUB_CACHE_DIR"] = "/users/mbourahl/.cache"


# Configure logging
logging.basicConfig(
    filename="model.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def load_ground_truth(gt_dir):
    # Get the ear data
    ear_data = pd.read_csv(gt_dir)

    # Convert all column names to lowercase
    ear_data.columns = map(str.lower, ear_data.columns)

    ear_data["tv"] = ear_data["tv"].replace(r"^\s*$", "0", regex=True)
    ear_data["tv"] = ear_data["tv"].fillna("0")
    ear_data["tv"] = ear_data["tv"].astype(int)

    # Keep only records where coders agree on "tv" column
    agreed_data = ear_data.groupby("filename").filter(lambda x: x["tv"].nunique() == 1)

    # Drop duplicates based on FileName, keep the first record
    agreed_data = agreed_data.drop_duplicates(subset="filename", keep="first")

    # Split the data into two groups based on the value of "tv"
    tv_0 = agreed_data[agreed_data["tv"] == 0]
    tv_1 = agreed_data[agreed_data["tv"] == 1]

    # Find out which group is larger
    larger_group = tv_0 if len(tv_0) > len(tv_1) else tv_1
    smaller_group = tv_1 if larger_group is tv_0 else tv_0

    # Randomly sample from the larger group to match the size of the smaller group
    larger_group = larger_group.sample(len(smaller_group), random_state=42)
    logging.info(f"Total data: {len(larger_group) + len(smaller_group)}")

    # Concatenate the balanced data
    balanced_data = pd.concat([larger_group, smaller_group])
    balanced_data.set_index("filename", inplace=True)

    return balanced_data


def split_data(balanced_data):
    # first split to train and temp
    balanced_data_train, balanced_data_temp = train_test_split(
        balanced_data, test_size=0.3, random_state=42
    )

    # second split to validation and test
    balanced_data_val, balanced_data_test = train_test_split(
        balanced_data_temp, test_size=0.5, random_state=42
    )

    return balanced_data_train, balanced_data_val, balanced_data_test


# LSTM Model
def lstm_model(input_shape):
    logging.info(f"Creating model with input shape {input_shape}")
    adam = optimizers.Adam(3e-4)
    inputs = Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(adam, "categorical_crossentropy", metrics=["accuracy"])

    return model


# CNN Model
def cnn_model(input_shape):
    logging.info(f"Creating model with input shape {input_shape}")
    adam = optimizers.Adam(3e-4)
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(128, kernel_size=3, activation="relu")(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(adam, "categorical_crossentropy", metrics=["accuracy"])

    return model


# Transformer Model
def transformer_model(input_dim):
    adam = optimizers.Adam(3e-4)
    # Define model parameters
    num_layers = 2
    embed_dim = input_dim[1]  # 1024
    num_heads = 2
    ff_dim = 512
    num_classes = 2
    input_shape = (None, embed_dim)

    # Initialize and compile model
    model = TransformerClassifier(
        num_layers, embed_dim, num_heads, ff_dim, input_shape, num_classes
    )
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluate_model(model, test_generator, output_path):
    logging.info("Evaluating model")
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

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    with open(f"{output_path}_results.json", "w") as json_file:
        json.dump(results, json_file)

    return results


def get_early_stopping_callback(patience=10):
    return tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=patience,
        verbose=0,
        mode="auto",
        restore_best_weights=True,
    )


def plot_history(history, output_dir, model_name):
    output_path = f"{output_dir}/{model_name}_history.png"
    plt.figure(figsize=(12, 8))

    # Plot the training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{model_name.upper()} Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper right")

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{model_name.upper()} Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{output_path}")


def train_model(
    model_func,
    args,
    training_generator,
    validation_generator=None,
):
    model_name = model_func.__name__.split("_")[0]

    # Callbacks
    early_stopping_callback = get_early_stopping_callback()

    # Get a batch of data
    data_batch, _ = training_generator.__getitem__(0)

    # Get the shape of a single sample
    input_shape = data_batch[0].shape
    model = model_func(input_shape)

    if args.deploy:
        model.fit(
            training_generator,
            epochs=args.epochs,
            callbacks=[early_stopping_callback],
        )
    else:
        history = model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=args.epochs,
            callbacks=[early_stopping_callback],
        )
        plot_history(history, args.output_dir, model_name)

    return model


def export_model(
    model_func,
    data_generator,
    args,
):
    logging.info("Exporting model...")
    model_name = model_func.__name__.split("_")[0]
    # Train on all data and save the model
    model = train_model(model_func, args, data_generator)

    path = f"{args.output_dir}/{model_name}.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    with tf.io.gfile.GFile(path, "wb") as f:
        f.write(tflite_model)


def initialize_args(parser):
    # Input paths

    # Add argument for number of epochs
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--deploy", action="store_true", default=False, help="Deploy model or not"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the directory containing NPY files",
    )
    parser.add_argument(
        "--gt_dir", required=True, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    balanced_data = load_ground_truth(args.gt_dir)

    if args.deploy:
        full_data_generator = DataGenerator(args.data_dir, balanced_data)

    else:
        balanced_data_train, balanced_data_val, balanced_data_test = split_data(
            balanced_data
        )
        training_generator = DataGenerator(args.data_dir, balanced_data_train)
        validation_generator = DataGenerator(args.data_dir, balanced_data_val)
        test_generator = DataGenerator(args.data_dir, balanced_data_test)

    models_func = [lstm_model, cnn_model, transformer_model]
    for model_func in models_func:
        if args.deploy:
            export_model(model_func, full_data_generator, args)
        else:
            model = train_model(
                model_func, args, training_generator, validation_generator
            )
            output_path = f"{args.output_dir}/{model_func.__name__.split('_')[0]}"
            evaluate_model(model, test_generator, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
