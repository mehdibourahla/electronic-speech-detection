import os
import json
import pandas as pd
import numpy as np

from data_generator import DataGenerator
from test import load_models, compute_metrics
import argparse


def load_test_set(directory_path):
    file_path = f"{directory_path}/test.csv"
    if os.path.exists(file_path):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        df.set_index("filename", inplace=True)
    else:
        print(f"File {file_path} does not exist")
        exit(1)
    return df


def bayesian_ensemble(models, X):
    num_models = len(models)
    num_classes = 2

    # Initialize ensemble predictions
    ensemble_predictions = np.zeros((X.shape[0], num_classes))

    for model in models:
        y_pred = model.predict(X)

        # Use the highest class probability as a confidence measure
        confidence = np.max(y_pred, axis=1)

        # Weight predictions by confidence
        weighted_predictions = y_pred * confidence[:, np.newaxis]

        # Add to ensemble predictions
        ensemble_predictions += weighted_predictions

    # Normalize the ensemble predictions
    ensemble_predictions /= num_models

    # Convert predictions from one-hot to labels

    ensemble_predictions = np.argmax(ensemble_predictions, axis=1)

    return ensemble_predictions


def initialize_args(parser):
    # Input paths

    parser.add_argument(
        "--gt_path",
        required=True,
        help="Path to the ground truth file (.csv)",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset to use (dse, aging, sins)",
    )

    parser.add_argument(
        "--ensemble_dir",
        required=True,
        help="Path to the ensemble directory",
    )

    parser.add_argument(
        "--output_dir", required=True, help="Path to the output directory"
    )


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.ensemble_dir.split("/")[-1]

    # Load data
    test = load_test_set(args.gt_path)
    test_generator = DataGenerator(test)
    X, y_test = test_generator.load_all_data()
    y_test = np.argmax(y_test, axis=1)

    # Load models
    input_shape = X.shape[1:]
    models = load_models(args.ensemble_dir, input_shape)
    # Compute Bayesian ensemble predictions
    y_pred = bayesian_ensemble(models, X)

    # Evaluate model
    results = dict(
        zip(
            ["accuracy", "precision", "recall", "f1", "specificity"],
            compute_metrics(y_pred, y_test),
        )
    )

    # Save results
    with open(f"{args.output_dir}/{model_name}_{args.dataset}.json", "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
