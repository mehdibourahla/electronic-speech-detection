import os
import json
import pandas as pd
from model_dse_aging import load_csv_files
from data_generator import DataGenerator
from model import evaluate_model, train_model, plot_history
import argparse


def initialize_args(parser):
    # Input paths

    # Add argument for number of epochs
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )

    parser.add_argument(
        "--fold_1", type=int, default=0, help="Fold number to train for"
    )
    parser.add_argument(
        "--fold_2", type=int, default=0, help="Fold number to train for"
    )

    parser.add_argument(
        "--gt_dir", default=None, help="Path to the ground truth CSV file"
    )

    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


def load_train_set(directory_path, fold_1, fold_2):
    file_path = f"{directory_path}/train_fold_{fold_1}_{fold_2}.csv"
    if os.path.exists(file_path):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
    else:
        print(f"File {file_path} does not exist")
        exit(1)
    return df


def main(args):
    args.output_dir = f"{args.output_dir}/fold_{args.fold_1}_{args.fold_2}"
    os.makedirs(args.output_dir, exist_ok=True)
    # Load data

    _, val, test = load_csv_files(args.gt_dir)
    train = load_train_set(args.gt_dir, args.fold_1, args.fold_2)
    train.set_index("filename", inplace=True)
    val.set_index("filename", inplace=True)
    test.set_index("filename", inplace=True)

    training_generator = DataGenerator(train)
    validation_generator = DataGenerator(val)
    test_generator = DataGenerator(test)

    # Train the model
    history, model = train_model(args, training_generator, validation_generator)
    plot_history(history, args.output_dir)

    # Evaluate the model
    results = evaluate_model(model, test_generator)
    with open(f"{args.output_dir}/test_results.json", "w") as json_file:
        json.dump(results, json_file)

    # Save the model weights as .h5
    path = f"{args.output_dir}/lstm.h5"
    model.save_weights(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
