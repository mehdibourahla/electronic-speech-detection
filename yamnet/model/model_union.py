import os
import json
import pandas as pd
from data_generator import DataGenerator
from model import evaluate_model, train_model, plot_history
from model_dse_aging import balance_classes, load_csv_files
import argparse


def merge_datasets(data_1, data_2, data_3):
    # Balance within each dataset
    data_1_balanced = balance_classes(data_1, "tv")
    data_2_balanced = balance_classes(data_2, "tv")
    data_3_balanced = balance_classes(data_3, "tv")

    # Balance between datasets
    min_size = min(
        data_1_balanced.shape[0], data_2_balanced.shape[0], data_3_balanced.shape[0]
    )
    data_1_final = data_1_balanced.sample(min_size, random_state=42)
    data_2_final = data_2_balanced.sample(min_size, random_state=42)
    data_3_final = data_3_balanced.sample(min_size, random_state=42)

    # Combine the balanced datasets
    final_balanced_data = pd.concat([data_1_final, data_2_final, data_3_final])
    final_balanced_data.set_index("filename", inplace=True)

    return final_balanced_data


def initialize_args(parser):
    # Input paths

    # Add argument for number of epochs
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )

    parser.add_argument(
        "--gt_dse", default=None, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--gt_aging", default=None, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--gt_sins", default=None, help="Path to the ground truth CSV file"
    )

    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data

    train_dse, val_dse, test_dse = load_csv_files(args.gt_dse)
    train_aging, val_aging, test_aging = load_csv_files(args.gt_aging)
    train_sins, val_sins, test_sins = load_csv_files(args.gt_sins)

    train = merge_datasets(train_dse, train_aging, train_sins)
    val = merge_datasets(val_dse, val_aging, val_sins)

    test_dse.set_index("filename", inplace=True)
    test_aging.set_index("filename", inplace=True)
    test_sins.set_index("filename", inplace=True)

    training_generator = DataGenerator(train)
    validation_generator = DataGenerator(val)

    test_dse_generator = DataGenerator(test_dse)
    test_aging_generator = DataGenerator(test_aging)
    test_sins_generator = DataGenerator(test_sins)

    # Train the model
    history, model = train_model(args, training_generator, validation_generator)
    plot_history(history, args.output_dir)

    # Evaluate the model
    results = evaluate_model(model, test_dse_generator)
    with open(f"{args.output_dir}/test_dse_results.json", "w") as json_file:
        json.dump(results, json_file)

    results = evaluate_model(model, test_aging_generator)
    with open(f"{args.output_dir}/test_aging_results.json", "w") as json_file:
        json.dump(results, json_file)

    results = evaluate_model(model, test_sins_generator)
    with open(f"{args.output_dir}/test_sins_results.json", "w") as json_file:
        json.dump(results, json_file)

    path = f"{args.output_dir}/lstm.h5"
    model.save_weights(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
