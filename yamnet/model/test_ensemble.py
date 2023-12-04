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


def ensemble(models, X, threshold=0.5):
    final_predictions = np.zeros(len(X))
    for clf in models:
        predictions = clf.predict(X)[:, 1]
        weights = np.where(predictions > threshold, predictions, 0)
        final_predictions += weights
    return np.round(final_predictions / len(models))


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
    y_pred = ensemble(models, X)

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
