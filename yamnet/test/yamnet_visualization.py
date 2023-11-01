import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

    # Split the data into two groups based on the value of "tv"
    tv_0 = agreed_data[agreed_data["tv"] == 0]
    tv_1 = agreed_data[agreed_data["tv"] == 1]

    # Find out which group is larger
    larger_group = tv_0 if len(tv_0) > len(tv_1) else tv_1
    smaller_group = tv_1 if larger_group is tv_0 else tv_0

    # Randomly sample from the larger group to match the size of the smaller group
    larger_group = larger_group.sample(len(smaller_group), random_state=42)

    # Concatenate the balanced data
    balanced_data = pd.concat([larger_group, smaller_group])
    balanced_data.set_index("filename", inplace=True)

    return balanced_data


def load_yamnet_features(data_dir, ground_truth):
    X = []
    y = []
    labels = ground_truth.tv.values
    list_IDs = ground_truth.index.tolist()
    for ID in list_IDs:
        try:
            sequence = np.load(f"{data_dir}/{ID}.npy", allow_pickle=True)
            X.append(sequence)
            y.append(labels[list_IDs.index(ID)])
        except Exception as e:
            continue

    X = pad_sequences(X, dtype="float32", padding="post")
    y = np.array(y)
    y = to_categorical(y)

    return X, y


def initialize_args(parser):
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the directory containing NPY files",
    )
    parser.add_argument(
        "--gt_path", required=True, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    ground_truth = load_ground_truth(args.gt_path)

    X, y = load_yamnet_features(args.data_dir, ground_truth)
    X_new = X.reshape(-1, 1024)
    y_new = [label for label in y for _ in range(31)]

    if not os.path.exists(os.path.join(args.output_dir, "X_TSNE.npy")):
        tsne = TSNE(n_components=2, verbose=1, random_state=42, n_jobs=-1)
        X_tsne = tsne.fit_transform(X_new)
        np.save(os.path.join(args.output_dir, "X_TSNE.npy"), X_tsne)
    else:
        X_tsne = np.load(os.path.join(args.output_dir, "X_TSNE.npy"))

    colors = ["red" if label == 1 else "green" for label in y_new]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=1, alpha=0.5)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(
        "YAMNet Embeddings Visualized with t-SNE (Red for TV and green for non-TV)"
    )
    plt.savefig(f"{args.output_dir}/yamnet_tsne.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
