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


def bayesian_ensemble(predictions, y):
    if len(y.shape) == 2 and y.shape[1] > 1:
        true_labels = y.argmax(axis=1)
    elif len(y.shape) == 1:
        true_labels = y
    else:
        raise ValueError(
            "y must be either a 1D array of class labels or a 2D one-hot encoded array."
        )

    true_labels = true_labels.astype(int)

    num_models = len(predictions)

    prior = 1.0 / num_models
    priors = np.full(num_models, prior)

    likelihoods = np.array(
        [pred[np.arange(len(y)), true_labels.astype(int)] for pred in predictions]
    )

    posteriors = priors[:, np.newaxis] * likelihoods

    posteriors = posteriors / posteriors.sum(axis=0, keepdims=True)

    weighted_predictions = np.array(predictions) * posteriors[:, :, np.newaxis]
    bayesian_avg_predictions = np.sum(weighted_predictions, axis=0)

    final_predictions = np.argmax(bayesian_avg_predictions, axis=1)

    return final_predictions


def ensemble_predictions(models, X, y):
    predictions = [model.predict(X) for model in models]

    # Bayesian ensemble
    bayesian_predictions = bayesian_ensemble(predictions, y)

    return bayesian_predictions


def load_models(model_dir):
    models = []
    for item in os.listdir(model_dir):
        model_path = os.path.join(model_dir, item)
        model = lstm_model((31, 1024))
        model.load_weights(model_path)
        models.append(model)
    return models


def load_ground_truth(gt_dir, data_dir, fold=1, step_ratio=0.5):
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

    # Split the data into two groups based on the value of "tv"
    tv_0 = agreed_data[agreed_data["tv"] == 0]
    tv_1 = agreed_data[agreed_data["tv"] == 1]

    # Find out which group is larger
    larger_group = tv_0 if len(tv_0) > len(tv_1) else tv_1
    smaller_group = tv_1 if larger_group is tv_0 else tv_0

    # Calculate step size for the sliding window approach
    step_size = int(len(smaller_group) * step_ratio)

    # Calculate the number of possible folds based on the step size
    num_folds = (len(larger_group) - len(smaller_group)) // step_size + 1

    # If the fold parameter is greater than the number of folds, set it to the last fold
    if fold > num_folds:
        print(f"Fold {fold} does not exist, using fold {num_folds}")
        fold = num_folds

    # Create the desired fold using the fold parameter
    start_idx = (fold - 1) * step_size
    end_idx = start_idx + len(smaller_group)
    selected_fold = larger_group.iloc[start_idx:end_idx]

    # Concatenate the balanced data
    balanced_data = pd.concat([selected_fold, smaller_group])
    balanced_data.set_index("filename", inplace=True)

    return balanced_data


def load_sins_gt(gt_dir, data_dir):
    gt_filemame = os.path.basename(gt_dir)
    print("Ground truth filename:", gt_filemame)

    df = pd.read_csv(gt_dir)

    # Replace "watching tv" with 1
    df.loc[df["activity"] == "watching tv", "activity"] = 1
    df.loc[df["activity"] != 1, "activity"] = 0

    df = df.rename(columns={"activity": "tv"})
    df = df.rename(columns={"audio_file": "filename"})

    # Get the list of .npy files in the directory
    files_in_dir = [
        os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".npy")
    ]
    print("Number of .npy files in directory:", len(files_in_dir))

    df = df[df["filename"].isin(files_in_dir)]
    print("Shape of GT after filtering by filenames:", df.shape)

    # Split the data into two groups based on the value of "tv"
    tv_0 = df[df["tv"] == 0]
    tv_1 = df[df["tv"] == 1]

    print("Number of records with tv=0:", len(tv_0))
    print("Number of records with tv=1:", len(tv_1))

    df.set_index("filename", inplace=True)

    return df


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
    parser.add_argument("--fold", type=int, default=1, help="Fold to use for testing")

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model (.h5) file)",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name",
    )

    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Whether to use ensemble predictions",
    )

    parser.add_argument(
        "--output_dir", required=True, help="Path to the output directory"
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1].split(".")[0]

    if args.dataset == "SINS":
        gt = load_sins_gt(args.gt_path, args.data_dir)
    elif args.dataset == "DSE" or args.dataset == "Aging":
        gt = load_ground_truth(args.gt_path, args.data_dir)
    else:
        raise ValueError("Dataset not supported")

    full_data_generator = DataGenerator(
        args.data_dir,
        gt,
    )

    X, y = full_data_generator.load_all_data()

    if args.ensemble:
        models = load_models(args.model)
        y_pred = ensemble_predictions(models, X, y[:, 1])
        results = compute_metrics(y_pred, y[:, 1])
        results = dict(
            zip(["accuracy", "precision", "recall", "f1", "specificity"], results)
        )
    else:
        results = model_evaluation(X, y, args.model)

    with open(f"{args.output_dir}/{model_name}.json", "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
