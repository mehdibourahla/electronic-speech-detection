import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from data_generator import DataGenerator
from model import evaluate_model, train_model, plot_history


def load_sins_gt(gt_dir, data_dir):
    df = pd.read_csv(gt_dir)

    df = df[~df["activity"].isin(["unknown", "dont use", "other"])]
    df["tv"] = df["activity"].apply(lambda x: 1 if x == "watching tv" else 0)
    df = df.rename(columns={"audio_file": "filename"})

    # Add the full path to the filename
    df["filename"] = df["filename"].apply(lambda x: f"{data_dir}/{x}.npy")

    tv_0 = df[df["tv"] == 0]
    tv_1 = df[df["tv"] == 1]

    working_absence_class = tv_0[
        (tv_0["activity"] == "absence") | (tv_0["activity"] == "working")
    ]
    non_working_absence_class = tv_0[
        (tv_0["activity"] != "absence") & (tv_0["activity"] != "working")
    ]

    downsampled_working_absence_class = working_absence_class.sample(
        len(tv_1) - len(non_working_absence_class), random_state=42
    )

    tv_0_balanced = pd.concat(
        [downsampled_working_absence_class, non_working_absence_class]
    )

    balanced_data = pd.concat([tv_0_balanced, tv_1])
    balanced_data.set_index("filename", inplace=True)

    print(f"Total data after excluding 'Unknown', 'Don't Use', 'Other': {len(df)}")
    print(f"Total balanced data: {len(balanced_data)}")

    return balanced_data


def split_data(balanced_data):
    # first split to train and temp
    balanced_data_train, balanced_data_temp = train_test_split(
        balanced_data, test_size=0.3, random_state=42, stratify=balanced_data["tv"]
    )

    # second split to validation and test
    balanced_data_val, balanced_data_test = train_test_split(
        balanced_data_temp,
        test_size=0.5,
        random_state=42,
        stratify=balanced_data_temp["tv"],
    )

    return balanced_data_train, balanced_data_val, balanced_data_test


def initialize_args(parser):
    # Input paths

    # Add argument for number of epochs
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )

    parser.add_argument(
        "--data_sins",
        required=True,
        help="Path to the directory containing NPY files",
    )
    parser.add_argument(
        "--gt_sins", required=True, help="Path to the ground truth CSV file"
    )

    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_sins_gt(args.gt_sins, args.data_sins)

    # Split the data into train, validation and test
    train, val, test = split_data(data)

    # Save the splits
    train.to_csv(f"{args.output_dir}/train.csv", index=False)
    val.to_csv(f"{args.output_dir}/val.csv", index=False)
    test.to_csv(f"{args.output_dir}/test.csv", index=False)

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
    path = f"{args.output_dir}/sins_lstm.h5"
    model.save_weights(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
