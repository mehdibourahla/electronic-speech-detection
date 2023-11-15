import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from data_generator import DataGenerator
from model import evaluate_model, train_model, plot_history

os.environ["TFHUB_CACHE_DIR"] = "/users/mbourahl/.cache"


# Configure logging
logging.basicConfig(
    filename="model.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

np.random.seed(42)
tf.random.set_seed(42)


def process_dataset(df, data_dir):
    files_in_dir = [
        os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".npy")
    ]

    df.columns = map(str.lower, df.columns)
    df = df[df["filename"].isin(files_in_dir)]

    df = df.rename(columns={"id": "participant_id"})
    df["tv"] = df["tv"].replace(r"^\s*$", 0, regex=True).fillna(0)
    df["problems"] = df["problems"].replace(r"^\s*$", 0, regex=True).fillna(0)
    df["tv"] = df["tv"].astype(int)
    df["problems"] = df["problems"].astype(int)
    df = df.groupby("filename").filter(lambda x: x["tv"].nunique() == 1)
    df = df.drop_duplicates(subset="filename", keep="first")
    df = df[df["problems"] == 0]

    return df


def load_process_data(gt_dir_1, gt_dir_2, data_dir_1, data_dir_2):
    # Load data
    data_1 = pd.read_csv(gt_dir_1)
    data_2 = pd.read_csv(gt_dir_2)

    # Process the dataset
    data_1 = process_dataset(data_1, data_dir_1)
    data_2 = process_dataset(data_2, data_dir_2)

    return data_1, data_2


def split_dataset(df, test_size=0.3, random_state=42):
    # Split participants
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_val_idx = next(gss.split(df, groups=df["participant_id"]))

    train_data = df.iloc[train_idx]
    test_val_data = df.iloc[test_val_idx]

    # Further split test_val_data into validation and test sets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_idx, test_idx = next(
        gss.split(test_val_data, groups=test_val_data["participant_id"])
    )

    validation_data = test_val_data.iloc[val_idx]
    test_data = test_val_data.iloc[test_idx]

    return train_data, validation_data, test_data


def balance_classes(data, class_column, random_state=42):
    """Balance classes within a dataset."""
    class_groups = data.groupby(class_column)
    min_class_size = class_groups.size().min()
    balanced_data = class_groups.apply(
        lambda x: x.sample(min_class_size, random_state=random_state)
    )
    return balanced_data


def merge_datasets(data_1, data_2, dir_1, dir_2):
    # Balance within each dataset
    data_1_balanced = balance_classes(data_1, "tv")
    data_2_balanced = balance_classes(data_2, "tv")

    # Balance between datasets
    min_size = min(data_1_balanced.shape[0], data_2_balanced.shape[0])
    data_1_final = data_1_balanced.sample(min_size, random_state=42)
    data_2_final = data_2_balanced.sample(min_size, random_state=42)

    print("Data 1 final shape:", data_1_final.shape)
    print("Data 2 final shape:", data_2_final.shape)

    # Add the full path to the filename
    data_1_final["filename"] = data_1_final["filename"].apply(
        lambda x: os.path.join(dir_1, x + ".npy")
    )
    data_2_final["filename"] = data_2_final["filename"].apply(
        lambda x: os.path.join(dir_2, x + ".npy")
    )

    print("Data DSE final shape:", data_1_final.shape)
    print("Data Aging final shape:", data_2_final.shape)

    print(
        "Number of unique participants in DSE:",
        data_1_final["participant_id"].nunique(),
    )
    print(
        "Number of unique participants in Aging:",
        data_2_final["participant_id"].nunique(),
    )

    print("Number of TV in DSE:", data_1_final["tv"].sum())
    print("Number of TV in Aging:", data_2_final["tv"].sum())

    # DSE Shape has to be equal to Aging Shape
    # DSE TV has to be equal to Aging TV

    # Combine the balanced datasets
    final_balanced_data = pd.concat([data_1_final, data_2_final])
    final_balanced_data.set_index("filename", inplace=True)

    print("Final balanced data shape:", final_balanced_data.shape)
    return final_balanced_data


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data_dse, data_aging = load_process_data(
        args.gt_dse, args.gt_aging, args.data_dse, args.data_aging
    )
    train_dse, val_dse, test_dse = split_dataset(data_dse)
    train_aging, val_aging, test_aging = split_dataset(data_aging)

    # Save the splits
    train_dse.to_csv(f"{args.output_dir}/train_dse.csv", index=False)
    val_dse.to_csv(f"{args.output_dir}/val_dse.csv", index=False)
    test_dse.to_csv(f"{args.output_dir}/test_dse.csv", index=False)
    train_aging.to_csv(f"{args.output_dir}/train_aging.csv", index=False)
    val_aging.to_csv(f"{args.output_dir}/val_aging.csv", index=False)
    test_aging.to_csv(f"{args.output_dir}/test_aging.csv", index=False)

    train = merge_datasets(train_dse, train_aging, args.data_dse, args.data_aging)
    val = merge_datasets(val_dse, val_aging, args.data_dse, args.data_aging)
    test = merge_datasets(test_dse, test_aging, args.data_dse, args.data_aging)

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
    path = f"{args.output_dir}/dse_aging_lstm.h5"
    model.save_weights(path)


def initialize_args(parser):
    # Input paths

    # Add argument for number of epochs
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )

    parser.add_argument(
        "--data_dse",
        required=True,
        help="Path to the directory containing NPY files",
    )
    parser.add_argument(
        "--gt_dse", required=True, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--data_aging",
        default=None,
        help="Path to the directory containing NPY files",
    )
    parser.add_argument(
        "--gt_aging", default=None, help="Path to the ground truth CSV file"
    )

    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
