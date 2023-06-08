import pandas as pd
import numpy as np
import logging
import json
import argparse


def load_ground_truth(gt_dir):
    # Get the ear data
    ear_data = pd.read_csv(gt_dir)

    # Convert all column names to lowercase
    ear_data.columns = map(str.lower, ear_data.columns)

    ear_data["tv"] = ear_data["tv"].replace(r"^\s*$", "0", regex=True)
    ear_data["tv"] = ear_data["tv"].fillna("0")
    ear_data["tv"] = ear_data["tv"].astype(int)

    # Keep only records where coders agree on "tv" column
    agreed_data = ear_data.groupby("filename").filter(lambda x: x["tv"].nunique() == 1)

    # Drop duplicates based on FileName, keep the first record
    agreed_data = agreed_data.drop_duplicates(subset="filename", keep="first")
    agreed_data.set_index("filename", inplace=True)
    return agreed_data


# Load and pad data
def load_and_pad_data(data_dir, ground_truth):
    data = []
    y = []
    for record in ground_truth.iterrows():
        try:
            sequence = np.load(f"{data_dir}/{record[0]}.npy", allow_pickle=True)
            y.append(record[1]["tv"])
            data.append(sequence)
        except Exception as e:
            logging.error(f"Error loading file {record[0]}.npy: {str(e)}")
            continue
    y = np.array(y)
    return data, y


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
        "--output_path", required=True, help="Path to the results file (.json)"
    )


def main(args):
    logging.info("Starting the main function...")

    ground_truth = load_ground_truth(args.gt_path)
    data, y = load_and_pad_data(args.data_dir, ground_truth)

    num_records = len(data)

    num_positive = sum([1 for label in y if label > 0])
    num_negative = sum([1 for label in y if label <= 0])

    results = {
        "num_records": num_records,
        "num_positive": num_positive,
        "num_negative": num_negative,
    }

    with open(args.output_path, "w") as json_file:
        json.dump(results, json_file)
    logging.info("Finished processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
