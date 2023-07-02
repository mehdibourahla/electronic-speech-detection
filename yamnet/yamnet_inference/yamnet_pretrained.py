import tensorflow_hub as hub
import tensorflow as tf
from pathlib import Path
import argparse
import os
import resampy
import numpy as np
import soundfile as sf
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


os.environ["TFHUB_CACHE_DIR"] = "/users/mbourahl/.cache"


# Configure logging
logging.basicConfig(
    filename="app.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def initialize_args(parser):
    # Input paths
    parser.add_argument(
        "--audio_dir",
        required=True,
        help="Path to the directory containing audio files",
    )
    parser.add_argument(
        "--gt_dir", required=True, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path to the output CSV file"
    )


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


def yamnet_tv(audio_wav, yamnet_model):
    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    class_names = list(pd.read_csv(class_map_path)["display_name"])

    scores, _, _ = yamnet_model(audio_wav)
    class_scores = tf.reduce_mean(scores, axis=0)
    tv_score, radio_score = [
        class_scores[class_names.index(name)].numpy()
        for name in ["Television", "Radio"]
    ]

    return {"tv": tv_score, "radio": radio_score}


def process_audio(path, ground_truth, yamnet_model):
    logging.info(f"Processing audio: {path}")
    try:
        wav_data, sr = sf.read(path, dtype=np.int16)
    except Exception as e:
        logging.error(f"Error loading audio file {path}: {e}")
        return None

    assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype("float32")

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != 16000:
        waveform = resampy.resample(waveform, sr, 16000)

    filename = path.split("/")[-1]
    # Check if the filename is in the ground truth dataframe
    if filename not in ground_truth.index.tolist():
        return None
    record = {"filename": filename, "isTV": ground_truth.loc[filename]["tv"]}

    yamnet_scores = yamnet_tv(waveform, yamnet_model)

    # Create a dictionary to store the results
    record["yamnet_tv"] = yamnet_scores["tv"]
    record["yamnet_radio"] = yamnet_scores["radio"]

    return record


def process_audio_wrapper(args):
    file, ground_truth, yamnet_model = args
    filename = str(file).split("/")[-1]
    if filename not in ground_truth.index:
        return None

    return process_audio(str(file), ground_truth, yamnet_model)


def process_audio_df(audio_dir, gt_dir, output_dir):
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    ground_truth = load_ground_truth(gt_dir)
    data_path = Path(audio_dir)
    data = list(data_path.glob("*.wav"))

    records = []
    processed_count = 0
    tv_count = 0

    with ThreadPoolExecutor() as executor:
        args_list = [(file, ground_truth, yamnet_model) for file in data]
        results = executor.map(process_audio_wrapper, args_list)

        # Log progress for each processed result
        for i, record in enumerate(results, start=1):
            if record is not None:
                records.append(record)
                processed_count += 1
                if record["isTV"] == 1:
                    tv_count += 1

    logging.info(
        f"Processed {processed_count}/{len(data)} audio files. Tv clips: {tv_count}"
    )

    df = pd.DataFrame(records)

    # Save the resulting dataframe to a CSV file
    df.to_csv(
        f"{output_dir}/processed_audio_data.csv",
        index=False,
    )


def main(args):
    logging.info("Starting the main function...")

    audio_dir = args.audio_dir
    gt_dir = args.gt_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    process_audio_df(audio_dir, gt_dir, output_dir)

    logging.info("Finished processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
