import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os


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

    # Split the data into two groups based on the value of "tv"
    tv_0 = agreed_data[agreed_data["tv"] == 0]
    tv_1 = agreed_data[agreed_data["tv"] == 1]

    # Find out which group is larger
    larger_group = tv_0 if len(tv_0) > len(tv_1) else tv_1
    smaller_group = tv_1 if larger_group is tv_0 else tv_0

    # Randomly sample from the larger group to match the size of the smaller group
    larger_group = larger_group.sample(len(smaller_group), random_state=42)
    logging.info(f"Total data: {len(larger_group) + len(smaller_group)}")

    # Concatenate the balanced data
    balanced_data = pd.concat([larger_group, smaller_group])
    balanced_data.set_index("filename", inplace=True)

    return balanced_data


def data_generation(data_dir, balanced_data):
    list_IDs = balanced_data.index.tolist()
    labels = balanced_data.tv.values

    X = []
    y = []

    # Generate data
    for ID in list_IDs:
        # Load sample and append to list
        try:
            sequence = np.load(f"{data_dir}/{ID}.npy", allow_pickle=True)
            X.append(sequence)

            # Store class
            y.append(labels[list_IDs.index(ID)])
        except Exception as e:
            continue

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")

    return X, y


def plot_distribution(X, y, output_dir):
    # Ensure that X and y are numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Convert arrays to DataFrame for seaborn
    df = pd.DataFrame({"feature": [x for x in X], "label": y})

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    sns.histplot(
        df[df["label"] == 0]["feature"],
        bins=10,
        kde=True,
        color="green",
        ax=axs[0],
    )
    axs[0].set_xlabel("Feature")
    axs[0].set_ylabel("Count")
    axs[0].set_title("Feature distribution histogram (Tv = 0)")

    sns.histplot(
        df[df["label"] == 1]["feature"],
        bins=10,
        kde=True,
        color="red",
        ax=axs[1],
    )
    axs[1].set_xlabel("Feature")
    axs[1].set_ylabel("Count")
    axs[1].set_title("Feature distribution histogram (Tv = 1)")

    plt.tight_layout()

    # Create 'output_dir' if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot to a file in 'output_dir'
    plt.savefig(f"{output_dir}/blue_histogram_plot.png")
    plt.close()


def load_features(path):
    data_path = Path(path)
    data = list(data_path.glob("*.npy"))
    X = []
    y = []
    for file in data:
        feature = np.load(file, allow_pickle=True)
        X.append(feature)

        filename = str(file).split("/")[-1]
        y.append(int(filename[1]))
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


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
        "--output_dir", required=True, help="Path to the output directory"
    )


def main(args):
    logging.info("Starting the main function...")
    os.makedirs(args.output_dir, exist_ok=True)
    balanced_data = load_ground_truth(args.gt_path)
    X, y = data_generation(args.data_dir, balanced_data)
    plot_distribution(X, y, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
