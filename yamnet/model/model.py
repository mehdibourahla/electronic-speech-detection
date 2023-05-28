import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import logging
import json


def initialize_args(parser):
    # Input paths
    parser.add_argument("--dataset", required=True, help="Name of the data used")
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
            y.append(record[1]["tv"])
            data.append(sequence)
        except Exception as e:
            logging.error(f"Error loading file {record[0]}.npy: {str(e)}")
            continue
    padded_data = pad_sequences(data, dtype="float32", padding="post")
    y = np.array(y)
    y = to_categorical(y)
    return padded_data, y


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


def evaluate_model(model, test_generator):
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

    return results


def train_model(
    model_func,
    training_generator,
    validation_generator,
    epochs=10,
):
    # Get a batch of data
    data_batch, _ = training_generator.__getitem__(0)

    # Get the shape of a single sample
    input_shape = data_batch[0].shape
    model = model_func(input_shape)

    model.fit(training_generator, validation_data=validation_generator, epochs=epochs)

    return model


def save_training_results(results, args):
    with open(
        f"{args.output_dir}/ear_{args.dataset}_{args.model_name}_results.json", "w"
    ) as json_file:
        json.dump(results, json_file)


def export_model(
    model_func,
    data_generator,
    args,
    epochs=10,
):
    logging.info("Exporting model...")

    # Train on all data and save the model
    # Get a batch of data
    data_batch, _ = data_generator.__getitem__(0)

    # Get the shape of a single sample
    input_shape = data_batch[0].shape
    model = model_func(input_shape)
    model.fit(data_generator, epochs=epochs)

    path = f"{args.output_dir}/{args.dataset}_{args.model_name}.tflite"
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
