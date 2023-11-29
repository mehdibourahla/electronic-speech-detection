import os
import argparse
import numpy as np
import soundfile as sf
import resampy
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import tensorflow as tf

import yamnet as yamnet_model
import params as yamnet_params


def yamnet_tv(waveform, yamnet):
    # Predict YAMNet classes.
    class_map_path = (
        tf.saved_model.Asset("yamnet_class_map.csv").asset_path.numpy().decode("utf-8")
    )
    class_names = list(pd.read_csv(class_map_path)["display_name"])
    waveform = np.reshape(waveform, [1, -1])
    scores, _, _, _ = yamnet.predict(waveform, steps=1, verbose=0)
    class_scores = tf.reduce_mean(scores, axis=0)

    tv_score, radio_score = [
        class_scores[class_names.index(name)].numpy()
        for name in ["Television", "Radio"]
    ]

    return tv_score + radio_score


def process_audio(audio_dir, output_dir, yamnet, params, duration=5):
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.wav"))

    # Open a text file to store the results
    with open(output_dir + "/yamnet_tv_results_v3.txt", "a") as results_file:
        for audio_file in tqdm(audio_files):
            try:
                wav_data, sr = sf.read(audio_file, dtype=np.int16)

                waveform = wav_data / 32768.0
                waveform = waveform.astype("float32")
                if len(waveform.shape) > 1:
                    waveform = np.mean(waveform, axis=1)
                if sr != params.sample_rate:
                    waveform = resampy.resample(waveform, sr, params.sample_rate)

                pace = int(duration * params.sample_rate)

                num_full_chunks = (
                    len(wav_data) // pace
                )  # Compute the number of full chunks

                for i in range(num_full_chunks):  # Loop only over full chunks
                    start_time = i * pace
                    end_time = start_time + pace
                    chunk = wav_data[start_time:end_time]

                    tv_score = yamnet_tv(chunk, yamnet)

                    # Write the filename and the tv value to the text file
                    results_file.write(f"{audio_file.stem}_{i}: {tv_score}\n")
            except Exception as e:
                print(f"Error Processing audio file {audio_file}: {e}")
                continue


def initialize_args(parser):
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the directory containing NPY files",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights("yamnet.h5")

    process_audio(args.data_dir, args.output_dir, yamnet, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
