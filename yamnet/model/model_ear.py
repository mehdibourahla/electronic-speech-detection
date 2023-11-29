import os
import json
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
        "--gt_dir", default=None, help="Path to the ground truth CSV file"
    )

    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data

    train, val, test = load_csv_files(args.gt_dir)
    train.set_index("filename", inplace=True)
    val.set_index("filename", inplace=True)
    test.set_index("filename", inplace=True)

    # Save the train, val, test splits as csv files
    # train.to_csv(f"{args.output_dir}/train.csv")
    # val.to_csv(f"{args.output_dir}/val.csv")
    # test.to_csv(f"{args.output_dir}/test.csv")

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
