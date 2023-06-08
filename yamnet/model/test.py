import numpy as np
import pandas as pd
import tensorflow as tf
import json
import argparse
import logging
from data_generator import DataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    filename="test.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def get_model(model_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def evaluate_model(model, test_generator):
    logging.info("Evaluating model")

    interpreter, input_details, output_details = get_model(model)

    y_pred = []
    y_test = []
    for i in range(len(test_generator)):
        x_batch, y_batch = test_generator[i]
        for x, y in zip(x_batch, y_batch):
            interpreter.set_tensor(input_details[0]["index"], [x])
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])

            y_pred.append(output_data[0][1])
            y_test.append(y)

    # Convert lists to arrays
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)

    # Convert predictions from one-hot to labels
    y_pred = (y_pred > 0.5).astype(int)
    if y_test.ndim > 1:
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

    agreed_data.set_index("filename", inplace=True)

    return agreed_data


def initialize_args(parser):
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the directory containing NPY files",
    )

    parser.add_argument(
        "--gt_path",
        required=True,
        help="Path to the ground truth file (.csv)",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model (.tflite) file)",
    )

    parser.add_argument(
        "--output_path", required=True, help="Path to the results file (.json)"
    )


def main(args):
    logging.info("Starting the main function...")

    ear_gt = load_ground_truth(args.gt_path)
    full_data_generator = DataGenerator(
        args.data_dir,
        ear_gt,
    )

    results = evaluate_model(args.model, full_data_generator)

    with open(args.output_path, "w") as json_file:
        json.dump(results, json_file)
    logging.info("Finished processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
