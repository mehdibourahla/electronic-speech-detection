import os
import json
import pandas as pd
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
from model import lstm_model


def load_ground_truth(gt_dir, data_dir):
    # Get the ear data
    gt_filemame = os.path.basename(gt_dir)
    print("Ground truth filename:", gt_filemame)

    ear_data = pd.read_csv(gt_dir)
    print("Shape of GT after reading:", ear_data.shape)

    # Get the list of .npy files in the directory
    files_in_dir = [
        os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".npy")
    ]
    print("Number of .npy files in directory:", len(files_in_dir))

    # Convert all column names to lowercase
    ear_data.columns = map(str.lower, ear_data.columns)

    # Filter data based on the filenames in the directories
    ear_data = ear_data[ear_data["filename"].isin(files_in_dir)]
    print("Shape of GT after filtering by filenames:", ear_data.shape)

    # Process the data
    ear_data["tv"] = ear_data["tv"].replace(r"^\s*$", "0", regex=True)
    ear_data["tv"] = ear_data["tv"].fillna("0")
    ear_data["tv"] = ear_data["tv"].astype(int)
    # Keep only records where coders agree on "tv" column
    agreed_data = ear_data.groupby("filename").filter(lambda x: x["tv"].nunique() == 1)
    # Drop duplicates based on FileName, keep the first record
    agreed_data = agreed_data.drop_duplicates(subset="filename", keep="first")
    print("Shape of GT after processing:", agreed_data.shape)

    agreed_data.set_index("filename", inplace=True)

    return agreed_data


def load_and_predict(model_func, weights_path, input_data):
    input_shape = input_data[0].shape

    model = model_func(input_shape)
    model.load_weights(weights_path)
    y_pred = model.predict(input_data)[:, 1]

    return y_pred


def compute_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    return accuracy, precision, recall, f1, specificity


def model_evaluation(X, y, model_path, threshold=0.5):
    y_pred = load_and_predict(lstm_model, model_path, X)
    y_pred_th = (y_pred > threshold).astype(int)
    y = y[:, 1]
    metrics = compute_metrics(y_pred_th, y)
    results = dict(
        zip(["accuracy", "precision", "recall", "f1", "specificity"], metrics)
    )

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
        help="Path to the model (.h5) file)",
    )

    parser.add_argument(
        "--output_dir", required=True, help="Path to the output directory"
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1].split(".")[0]

    ear_gt = load_ground_truth(args.gt_path, args.data_dir)
    full_data_generator = DataGenerator(
        args.data_dir,
        ear_gt,
    )

    X, y = full_data_generator.load_all_data()

    results = model_evaluation(X, y, args.model)

    with open(f"{args.output_dir}/{model_name}.json", "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
