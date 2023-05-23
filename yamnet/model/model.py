import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, optimizers, Input, Model
from transformer import TransformerClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import argparse
import logging
import json

os.environ["TFHUB_CACHE_DIR"] = "/users/mbourahl/.cache"


# Configure logging
logging.basicConfig(
    filename="model.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    logging.info(f"Found {len(physical_devices)} GPUs: {physical_devices}")
else:
    logging.warning("No GPUs found")


def load_ground_truth(gt_dir):
    # Get the ear data
    ear_data = pd.read_csv(gt_dir)

    ear_data["TV"] = ear_data["TV"].replace(r"^\s*$", "0", regex=True)
    ear_data["TV"] = ear_data["TV"].fillna("0")
    ear_data["TV"] = ear_data["TV"].astype(int)

    # Keep only records where coders agree on "Tv" column
    agreed_data = ear_data.groupby("FileName").filter(lambda x: x["TV"].nunique() == 1)

    # Drop duplicates based on FileName, keep the first record
    agreed_data = agreed_data.drop_duplicates(subset="FileName", keep="first")

    # Split the data into two groups based on the value of "Tv"
    tv_0 = agreed_data[agreed_data["TV"] == 0]
    tv_1 = agreed_data[agreed_data["TV"] == 1]

    # Find out which group is larger
    larger_group = tv_0 if len(tv_0) > len(tv_1) else tv_1
    smaller_group = tv_1 if larger_group is tv_0 else tv_0

    # Randomly sample from the larger group to match the size of the smaller group
    larger_group = larger_group.sample(len(smaller_group), random_state=42)
    logging.info(f"Total data: {len(larger_group) + len(smaller_group)}")
    # Concatenate the balanced data
    balanced_data = pd.concat([larger_group, smaller_group])
    balanced_data.set_index("FileName", inplace=True)

    return balanced_data


# Load pickle files
def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        data = np.load(f, allow_pickle=True)
    return data


# Load and pad data
def load_and_pad_data(data_dir, balanced_data):
    data = []
    y = []
    for record in balanced_data.iterrows():
        try:
            sequence = np.load(f"{data_dir}/{record[0]}.npy", allow_pickle=True)
            y.append(record[1]["TV"])
            data.append(sequence)
        except Exception as e:
            logging.error(f"Error loading file {record[0]}.npy: {str(e)}")
            continue
    padded_data = pad_sequences(data, dtype="float32", padding="post")
    y = np.array(y)
    y = to_categorical(y)
    return padded_data, y


def split_data(data, labels, train_size=0.7, validation_size=0.15):
    # Calculate the test size
    test_size = 1 - train_size

    # Split the data into training and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, train_size=train_size, test_size=test_size, random_state=42
    )

    # Calculate the ratio of validation data to the temporary data
    validation_ratio = validation_size / test_size

    # Split the temporary data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=validation_ratio,
        test_size=1 - validation_ratio,
        random_state=42,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


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


def transformer_model(input_dim):
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
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating model")
    # Predict on validation set
    y_pred = model.predict(X_test)

    # Convert predictions from one-hot to labels
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


def initialize_args(parser):
    # Input paths
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


def export_model(model, name, dir):
    logging.info("Exporting model...")

    path = f"{dir}/{name}.tflite"
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


def train_evaluate_save_model(
    model_func,
    model_name,
    input_shape,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    data,
    labels,
    output_dir,
    epochs=10,
    batch_size=32,
):
    # Train and evaluate
    model = model_func(input_shape)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
    )
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

    # Save results to a JSON file
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    with open(f"{output_dir}/ear_aging_{model_name}_results.json", "w") as json_file:
        json.dump(results, json_file)

    # Train on all data and save the model
    model = model_func(input_shape)
    model.fit(data, labels, epochs=epochs, batch_size=batch_size)
    export_model(model, f"ear_aging_{model_name}", output_dir)


def main(args):
    logging.info("Starting the main function...")

    balanced_data = load_ground_truth(args.gt_dir)
    data, labels = load_and_pad_data(args.data_dir, balanced_data)

    # Reshaping YAMNet features to 31 frames
    data = np.mean(data.reshape((-1, 31, 2, 1024)), axis=2)

    # Splitting data into train, validation and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(data, labels)

    input_shape = (X_train.shape[1], X_train.shape[2])

    train_evaluate_save_model(
        lstm_model,
        "lstm",
        input_shape,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        data,
        labels,
        args.output_dir,
    )
    train_evaluate_save_model(
        cnn_model,
        "cnn",
        input_shape,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        data,
        labels,
        args.output_dir,
    )
    train_evaluate_save_model(
        transformer_model,
        "transformer",
        input_shape,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        data,
        labels,
        args.output_dir,
    )

    logging.info("Finished processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
