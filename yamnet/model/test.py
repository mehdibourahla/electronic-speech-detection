import os
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from model import load_ground_truth

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
        "--output_dir", required=True, help="Path to the output directory"
    )


def main(args):
    logging.info("Starting the main function...")
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1].split(".")[0]

    ear_gt = load_ground_truth(args.gt_path)
    full_data_generator = DataGenerator(
        args.data_dir,
        ear_gt,
    )

    results = evaluate_model(args.model, full_data_generator)

    with open(f"{args.output_dir}/{model_name}.json", "w") as json_file:
        json.dump(results, json_file)
    logging.info("Finished processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
