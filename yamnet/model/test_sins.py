import os
import json
import argparse
import pandas as pd
from model import lstm_model, evaluate_model
from data_generator import DataGenerator


def load_test_set(directory_path):
    file_path = f"{directory_path}/test.csv"
    if os.path.exists(file_path):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
    else:
        print(f"File {file_path} does not exist")
        exit(1)
    return df


def load_model(weights_path, test_generator):
    data_batch, _ = test_generator.__getitem__(0)
    input_shape = data_batch[0].shape

    model = lstm_model(input_shape)
    model.load_weights(weights_path)

    return model


def initialize_args(parser):
    parser.add_argument("--node", required=True, help="SINS node to use")
    parser.add_argument(
        "--gt_dir",
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
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    test = load_test_set(args.gt_dir)
    test.set_index("filename", inplace=True)
    test_generator = DataGenerator(test)

    # Load model
    model = load_model(args.model, test_generator)

    # Evaluate the model
    results = evaluate_model(model, test_generator)
    with open(f"{args.output_dir}/Node_{args.node}_results.json", "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
