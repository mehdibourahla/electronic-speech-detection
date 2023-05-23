import numpy as np
import pandas as pd
import tensorflow as tf
import json
import argparse
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    filename="test.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def get_model(model_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def predict(model, X):
    interpreter, input_details, output_details = get_model(model)

    interpreter.set_tensor(input_details[0]["index"], [X])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data[0][1]


def evaluate_model(y_pred, y_test):
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


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
        "--model",
        required=True,
        help="Path to the model (.tflite) file)",
    )

    parser.add_argument(
        "--output_path", required=True, help="Path to the results file (.json)"
    )


def load_ground_truth(gt_dir):
    # Get the ear data
    ear_data = pd.read_csv(gt_dir)

    ear_data["Tv"] = ear_data["Tv"].replace(r"^\s*$", "0", regex=True)
    ear_data["Tv"] = ear_data["Tv"].fillna("0")

    ear_data["Tv"] = ear_data["Tv"].astype(int)

    # Drop duplicates based on FileName, keep the first record
    ear_data = ear_data.drop_duplicates(subset="FileName", keep="first")
    ear_data.set_index("FileName", inplace=True)

    return ear_data


# Load and pad data
def load_and_pad_data(data_dir, gt_data):
    data = []
    y = []
    for record in gt_data.iterrows():
        try:
            sequence = np.load(f"{data_dir}/{record[0]}.npy", allow_pickle=True)
            y.append(record[1]["Tv"])
            data.append(sequence)
        except Exception as e:
            logging.error(f"Error loading file {record[0]}.npy: {str(e)}")
            continue
    padded_data = pad_sequences(data, dtype="float32", padding="post")
    y = np.array(y)
    return padded_data, y


def main(args):
    logging.info("Starting the main function...")

    ear_gt = load_ground_truth(args.gt_path)
    data, labels = load_and_pad_data(args.data_dir, ear_gt)

    # Reshaping YAMNet features to 31 frames
    data = np.mean(data.reshape((-1, 31, 2, 1024)), axis=2)
    y_pred = np.array([predict(args.model, X) for X in data])

    y_pred_th = (y_pred > 0.5).astype(int)
    accuracy, precision, recall, f1 = evaluate_model(y_pred_th, labels)

    # Save results to a JSON file
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    with open(args.output_path, "w") as json_file:
        json.dump(results, json_file)
    logging.info("Finished processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
