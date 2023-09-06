import tensorflow as tf
import argparse
import os
import numpy as np
from data_generator import DataGenerator
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import json

os.environ["TFHUB_CACHE_DIR"] = "/users/mbourahl/.cache"


def load_ground_truth(gt_dir):
    # Get the ear data
    df = pd.read_csv(gt_dir)

    # Replace "watching tv" with 1
    df.loc[df["activity"] == "watching tv", "activity"] = 1

    # Replace all other activities with 0
    df.loc[df["activity"] != 1, "activity"] = 0

    df = df.rename(columns={"activity": "tv"})
    df = df.rename(columns={"audio_file": "filename"})
    # Split the data into two groups based on the value of "tv"
    tv_0 = df[df["tv"] == 0]
    tv_1 = df[df["tv"] == 1]

    # Find out which group is larger
    larger_group = tv_0 if len(tv_0) > len(tv_1) else tv_1
    smaller_group = tv_1 if larger_group is tv_0 else tv_0

    # Randomly sample from the larger group to match the size of the smaller group
    larger_group = larger_group.sample(len(smaller_group), random_state=42)
    print(f"Total data: {len(larger_group) + len(smaller_group)}")

    # Concatenate the balanced data
    balanced_data = pd.concat([larger_group, smaller_group])
    balanced_data.set_index("filename", inplace=True)

    return balanced_data


def get_model(model_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def evaluate_model(model, test_generator):
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
    # Input paths
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the directory containing YAMNet embeddings files",
    )
    parser.add_argument(
        "--gt_path", required=True, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the model (.tflite) file)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path to the output CSV file"
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset the model has been trained on"
    )


def main(args):
    model_path = args.model_path
    data_dir = args.data_dir
    gt_path = args.gt_path
    output_dir = args.output_dir
    dataset = args.dataset

    model_name = model_path.split("/")[-1].split(".")[0]
    sins_gt = load_ground_truth(gt_path)
    full_data_generator = DataGenerator(
        data_dir,
        sins_gt,
    )

    os.makedirs(output_dir, exist_ok=True)
    results = evaluate_model(model_path, full_data_generator)

    with open(f"{output_dir}/{dataset}_{model_name}.json", "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
