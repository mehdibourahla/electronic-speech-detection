import tensorflow_hub as hub
import tensorflow as tf
from pathlib import Path
import argparse
import os
import librosa
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


def get_ear_data(path):
    # Get the ear data
    ear_data = pd.read_csv(path)
    ear_data["Tv"] = ear_data["Tv"].astype(int)

    # Log the number of records before filtering
    logging.info(f"Number of records before filtering: {len(ear_data)}")

    # Group by FileName and keep only records where coders agree on "Tv" column
    agreed_data = ear_data.groupby("FileName").filter(lambda x: x["Tv"].nunique() == 1)

    # Log the number of records after filtering
    logging.info(f"Number of records after filtering: {len(agreed_data)}")

    # Set the index to be the FileName
    agreed_data.set_index("FileName", inplace=True)
    return agreed_data


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
        audio_wav, _ = librosa.load(path, sr=16000)
    except Exception as e:
        logging.error(f"Error loading audio file {path}: {e}")
        return None

    filename = path.split("/")[-1]
    record_series = ground_truth.loc[filename].iloc[0]
    record = record_series.to_dict()
    record["FileName"] = record_series.name

    yamnet_scores = yamnet_tv(audio_wav, yamnet_model)
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
    ground_truth = get_ear_data(gt_dir)
    data_path = Path(audio_dir)
    data = list(data_path.glob("*.wav"))

    records = []
    processed_count = 0
    tv_count = 0  # Add a variable to keep track of Tv labeled audio clips

    with ThreadPoolExecutor(max_workers=16) as executor:
        args_list = [(file, ground_truth, yamnet_model) for file in data]
        results = executor.map(process_audio_wrapper, args_list)

        # Log progress for each processed result
        for i, record in enumerate(results, start=1):
            if record is not None:
                records.append(record)
                processed_count += 1
                if record["Tv"] == 1:  # Update the tv_count if Tv label is 1
                    tv_count += 1
            logging.info(
                f"Processed {i}/{len(data)} audio files. Successful: {processed_count}. Tv clips: {tv_count}"
            )

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

    process_audio_df(audio_dir, gt_dir, output_dir)

    logging.info("Finished processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
