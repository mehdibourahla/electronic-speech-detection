import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, Input, Model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator, DataGeneratorMultiple

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


def load_balanced_data(
    gt_dir_1, gt_dir_2, data_dir_1, data_dir_2, fold_1, fold_2, step_ratio=0.5
):
    # Load data
    data_1 = pd.read_csv(gt_dir_1)
    data_2 = pd.read_csv(gt_dir_2)

    print(f"Ground truth filename: {os.path.basename(gt_dir_1)} Fold {fold_1}")
    print(f"Ground truth filename: {os.path.basename(gt_dir_2)} Fold {fold_2}")

    print("Shape of data_1 after reading:", data_1.shape)
    print("Shape of data_2 after reading:", data_2.shape)

    # Get the list of .npy files in the respective directories
    files_in_dir_1 = [
        os.path.splitext(f)[0] for f in os.listdir(data_dir_1) if f.endswith(".npy")
    ]
    files_in_dir_2 = [
        os.path.splitext(f)[0] for f in os.listdir(data_dir_2) if f.endswith(".npy")
    ]

    print("Number of .npy files in dir_1:", len(files_in_dir_1))
    print("Number of .npy files in dir_2:", len(files_in_dir_2))

    # Convert all column names to lowercase
    data_1.columns = map(str.lower, data_1.columns)
    data_2.columns = map(str.lower, data_2.columns)

    # Filter data based on the filenames in the directories
    data_1 = data_1[data_1["filename"].isin(files_in_dir_1)]
    data_2 = data_2[data_2["filename"].isin(files_in_dir_2)]

    print("Shape of data_1 after filtering by filenames:", data_1.shape)
    print("Shape of data_2 after filtering by filenames:", data_2.shape)

    # Process both datasets
    data_list = [data_1, data_2]
    for i, data in enumerate(data_list):
        data_list[i]["tv"] = data["tv"].replace(r"^\s*$", "0", regex=True)
        data_list[i]["tv"] = data["tv"].fillna("0")
        data_list[i]["tv"] = data["tv"].astype(int)
        data_list[i] = data.groupby("filename").filter(lambda x: x["tv"].nunique() == 1)
        data_list[i] = data.drop_duplicates(subset="filename", keep="first")

    # Re-assign the modified data back to data_1 and data_2
    data_1, data_2 = data_list

    print("Shape of data_1 after processing:", data_1.shape)
    print("Shape of data_2 after processing:", data_2.shape)

    # Firstly, balance within each dataset
    balanced_data_list = []
    for idx, data in enumerate(data_list):
        tv_0 = data[data["tv"] == 0][["filename", "tv"]]
        tv_1 = data[data["tv"] == 1][["filename", "tv"]]
        larger_group = tv_0 if len(tv_0) > len(tv_1) else tv_1
        smaller_group = tv_1 if larger_group is tv_0 else tv_0

        step_size = int(len(smaller_group) * step_ratio)
        fold = fold_1 if idx == 0 else fold_2

        start_idx = (fold - 1) * step_size
        end_idx = start_idx + len(smaller_group)
        selected_fold = larger_group.iloc[start_idx:end_idx]

        print(f"Dataset {idx+1} larger group size:", len(larger_group))
        print(f"Dataset {idx+1} smaller group size:", len(smaller_group))

        balanced_data = pd.concat([selected_fold, smaller_group])
        balanced_data_list.append(balanced_data)

    # Now balance between datasets
    min_tv_0 = min(
        len(balanced_data_list[0][balanced_data_list[0]["tv"] == 0]),
        len(balanced_data_list[1][balanced_data_list[1]["tv"] == 0]),
    )
    min_tv_1 = min(
        len(balanced_data_list[0][balanced_data_list[0]["tv"] == 1]),
        len(balanced_data_list[1][balanced_data_list[1]["tv"] == 1]),
    )

    print("Minimum tv_0 count between datasets:", min_tv_0)
    print("Minimum tv_1 count between datasets:", min_tv_1)

    dir_mapping = {}
    # Downsample both datasets to the minimum counts
    for i in range(len(balanced_data_list)):
        tv_0_sample = balanced_data_list[i][balanced_data_list[i]["tv"] == 0].sample(
            min_tv_0, random_state=42
        )
        tv_1_sample = balanced_data_list[i][balanced_data_list[i]["tv"] == 1].sample(
            min_tv_1, random_state=42
        )
        balanced_data_list[i] = pd.concat([tv_0_sample, tv_1_sample])
        balanced_data_list[i].set_index("filename", inplace=True)

        print(
            f"Dataset {i+1} after balancing between datasets:",
            balanced_data_list[i].shape,
        )

        dir_mapping.update(
            {
                filename: data_dir_1 if i == 0 else data_dir_2
                for filename in balanced_data_list[i].index
            }
        )

    # Combine the balanced datasets
    final_balanced_data = pd.concat(balanced_data_list)

    print("Final balanced data shape:", final_balanced_data.shape)
    return final_balanced_data, dir_mapping


def load_ground_truth(gt_dir, data_dir, fold=1, step_ratio=0.5):
    # Get the ear data
    gt_filemame = os.path.basename(gt_dir)
    print("Ground truth filename:", gt_filemame)

    ear_data = pd.read_csv(gt_dir)
    print("Shape of GT after reading:", ear_data.shape)

    # Get the list of .npy files in the directory
    files_in_dir = [
        os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".npy")
    ]
    print("Number of .npy files in directory:", len(files_in_dir))

    # Convert all column names to lowercase
    ear_data.columns = map(str.lower, ear_data.columns)

    # Filter data based on the filenames in the directories
    ear_data = ear_data[ear_data["filename"].isin(files_in_dir)]
    print("Shape of GT after filtering by filenames:", ear_data.shape)

    # Process the data
    ear_data["tv"] = ear_data["tv"].replace(r"^\s*$", "0", regex=True)
    ear_data["tv"] = ear_data["tv"].fillna("0")
    ear_data["tv"] = ear_data["tv"].astype(int)
    # Keep only records where coders agree on "tv" column
    agreed_data = ear_data.groupby("filename").filter(lambda x: x["tv"].nunique() == 1)
    # Drop duplicates based on FileName, keep the first record
    agreed_data = agreed_data.drop_duplicates(subset="filename", keep="first")
    print("Shape of GT after processing:", agreed_data.shape)

    # Split the data into two groups based on the value of "tv"
    tv_0 = agreed_data[agreed_data["tv"] == 0]
    tv_1 = agreed_data[agreed_data["tv"] == 1]

    # Find out which group is larger
    larger_group = tv_0 if len(tv_0) > len(tv_1) else tv_1
    smaller_group = tv_1 if larger_group is tv_0 else tv_0

    # Calculate step size for the sliding window approach
    step_size = int(len(smaller_group) * step_ratio)

    # Calculate the number of possible folds based on the step size
    num_folds = (len(larger_group) - len(smaller_group)) // step_size + 1

    # If the fold parameter is greater than the number of folds, set it to the last fold
    if fold > num_folds:
        print(f"Fold {fold} does not exist, using fold {num_folds}")
        fold = num_folds

    # Create the desired fold using the fold parameter
    start_idx = (fold - 1) * step_size
    end_idx = start_idx + len(smaller_group)
    selected_fold = larger_group.iloc[start_idx:end_idx]

    # Concatenate the balanced data
    balanced_data = pd.concat([selected_fold, smaller_group])
    balanced_data.set_index("filename", inplace=True)

    return balanced_data


def split_data(balanced_data):
    # first split to train and temp
    balanced_data_train, balanced_data_temp = train_test_split(
        balanced_data, test_size=0.3, random_state=42
    )

    # second split to validation and test
    balanced_data_val, balanced_data_test = train_test_split(
        balanced_data_temp, test_size=0.5, random_state=42
    )

    return balanced_data_train, balanced_data_val, balanced_data_test


# LSTM Model
def lstm_model(input_shape):
    adam = optimizers.Adam(3e-4)
    inputs = Input(shape=input_shape)

    # Using glorot_uniform (Xavier) for LSTM and he_normal for Dense with softmax
    x = layers.LSTM(
        64,
        return_sequences=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
    )(inputs)
    x = layers.LSTM(
        64, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal"
    )(x)
    outputs = layers.Dense(
        2, activation="softmax", kernel_initializer="glorot_uniform"
    )(x)

    model = Model(inputs, outputs)
    model.compile(adam, "categorical_crossentropy", metrics=["accuracy"])

    return model


# CNN Model
def cnn_model(input_shape):
    logging.info(f"Creating CNN model with input shape {input_shape}")
    adam = optimizers.Adam(3e-4)
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(128, kernel_size=3, activation="relu")(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(adam, "categorical_crossentropy", metrics=["accuracy"])

    return model


def evaluate_model(model, test_generator, output_path):
    y_pred = []
    y_test = []
    for i in range(len(test_generator)):
        x, y = test_generator[i]
        y_pred.extend(model.predict(x))
        y_test.extend(y)

    # Convert lists to arrays
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)

    # Convert predictions from one-hot to labels
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }

    with open(f"{output_path}_results.json", "w") as json_file:
        json.dump(results, json_file)

    return results


def get_early_stopping_callback(monitor="val_loss", patience=10):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=0,
        patience=patience,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )
    return early_stopping_callback


def plot_history(history, output_dir, model_name):
    output_path = f"{output_dir}/{model_name}_history.png"
    plt.figure(figsize=(12, 8))

    # Plot the training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{model_name.upper()} Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper right")

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{model_name.upper()} Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{output_path}")


def train_model(
    model_func,
    args,
    training_generator,
    validation_generator=None,
):
    model_name = model_func.__name__.split("_")[0]

    # Callbacks
    early_stopping_callback = get_early_stopping_callback(monitor="val_loss")

    # Get a batch of data
    data_batch, _ = training_generator.__getitem__(0)

    # Get the shape of a single sample
    input_shape = data_batch[0].shape
    model = model_func(input_shape)

    history = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=args.epochs,
        callbacks=[early_stopping_callback],
    )

    # Log the epoch at which training was stopped
    logging.info(
        f"Training for {model_name} stopped at epoch {early_stopping_callback.stopped_epoch}"
    )
    return history, model


def export_model(
    model_func,
    training_generator,
    validation_generator,
    args,
):
    model_name = model_func.__name__.split("_")[0]
    history, model = train_model(
        model_func, args, training_generator, validation_generator
    )

    plot_history(history, args.output_dir, model_name)

    # Save the model weights as .h5
    path = f"{args.output_dir}/{model_name}_v2.h5"
    model.save_weights(path)


def initialize_args(parser):
    # Input paths

    # Add argument for number of epochs
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--deploy", action="store_true", default=False, help="Deploy model or not"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to use for training or deployment",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the directory containing NPY files",
    )
    parser.add_argument(
        "--gt_dir", required=True, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--data_dir2",
        default=None,
        help="Path to the directory containing NPY files",
    )
    parser.add_argument(
        "--gt_dir2", default=None, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--fold_1", type=int, default=1, help="Fold to use for training"
    )
    parser.add_argument(
        "--fold_2", type=int, default=1, help="Fold to use for training"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    if args.data_dir2 and args.gt_dir2:
        print("Loading data from two directories")
        balanced_data, dir_mapping = load_balanced_data(
            args.gt_dir,
            args.gt_dir2,
            args.data_dir,
            args.data_dir2,
            args.fold_1,
            args.fold_2,
        )
        generator_class = DataGeneratorMultiple
        generator_args = (dir_mapping,)
    else:
        balanced_data = load_ground_truth(args.gt_dir, args.data_dir, args.fold_1)
        generator_class = DataGenerator
        generator_args = (args.data_dir,)

    # Generate data
    if args.deploy:
        balanced_data_train, balanced_data_val = train_test_split(
            balanced_data, test_size=0.2, random_state=42
        )
        training_generator = generator_class(*generator_args, balanced_data_train)
        validation_generator = generator_class(*generator_args, balanced_data_val)
    else:
        split_data_list = split_data(balanced_data)
        data_generators = [
            generator_class(*generator_args, data) for data in split_data_list
        ]
        training_generator, validation_generator, test_generator = data_generators

    # Define models
    models_map = {
        "lstm": lstm_model,
        "cnn": cnn_model,
    }
    model_func = models_map[args.model]

    if args.deploy:
        logging.info(f"### Exporting {model_func.__name__} model ###")
        export_model(model_func, training_generator, validation_generator, args)
    else:
        logging.info(f"### Training {model_func.__name__} model ###")
        _, model = train_model(
            model_func, args, training_generator, validation_generator
        )
        output_path = f"{args.output_dir}/{model_func.__name__.split('_')[0]}"
        evaluate_model(model, test_generator, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
