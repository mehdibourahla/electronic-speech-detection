import os
from pathlib import Path
import argparse
import logging
import numpy as np
import yamnet as yamnet_model
import params as yamnet_params
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
import resampy


# Configure logging
logging.basicConfig(
    filename="features.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def yamnet_embeddings(waveform, yamnet):
    # Predict YAMNet classes.
    _, embeddings_final, _, _ = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
    # Shape: (N, 1024).
    return embeddings_final


def process_audio(path, output_dir, yamnet, params):
    logging.info(f"Processing audio: {path}")
    try:
        wav_data, sr = sf.read(path, dtype=np.int16)

        assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        waveform = waveform.astype("float32")

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if sr != params.sample_rate:
            waveform = resampy.resample(waveform, sr, params.sample_rate)

        duration = 30  # Duration in seconds
        pace = int(duration * params.sample_rate)

        num_full_chunks = len(waveform) // pace

        filename = path.split("/")[-1].split(".")[0]
        for i in range(num_full_chunks):
            start_time = i * pace
            end_time = start_time + pace
            chunk = wav_data[start_time:end_time]
            embeddings = yamnet_embeddings(chunk, yamnet)
            embeddings_file = f"{output_dir}/{filename}_{i+1}.wav.npy"
            np.save(embeddings_file, embeddings)

    except Exception as e:
        print(f"Error Processing audio file {path}: {e}")
        return None


def process_audio_wrapper(audio_dir, output_dir, yamnet, params):
    data_path = Path(audio_dir)
    data = list(data_path.glob("*.wav"))
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_audio, str(file), output_dir, yamnet, params)
            for file in data
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                logging.error(f"Error occurred: {e}")


def main(args):
    logging.info("Starting the main function...")

    audio_dir = args.audio_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights("yamnet.h5")

    process_audio_wrapper(audio_dir, output_dir, yamnet, params)

    logging.info("Finished processing.")


def initialize_args(parser):
    # Input paths
    parser.add_argument(
        "--audio_dir",
        required=True,
        help="Path to the directory containing audio files",
    )

    parser.add_argument(
        "--output_dir", required=True, help="Path to the output NPY files"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
