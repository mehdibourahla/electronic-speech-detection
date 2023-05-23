import tensorflow_hub as hub
from pathlib import Path
import argparse
import librosa
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    filename="features.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def yamnet_embeddings(audio_wav, yamnet_model):
    _, embeddings, _ = yamnet_model(audio_wav)
    # Shape: (N, 1024).
    return embeddings


def process_audio(path, output_dir, yamnet_model):
    logging.info(f"Processing audio: {path}")
    try:
        audio_wav, _ = librosa.load(path, sr=16000)
    except Exception as e:
        logging.error(f"Error loading audio file {path}: {e}")
        return None

    filename = path.split("/")[-1]

    embeddings = yamnet_embeddings(audio_wav, yamnet_model)
    # Save YAMNet embeddings for the audio file
    embeddings_file = f"{output_dir+'/'+filename}.npy"
    try:
        np.save(embeddings_file, embeddings)
    except Exception as e:
        logging.error(f"Error saving embeddings for {filename}: {e}")
        return None


def process_audio_wrapper(audio_dir, output_dir):
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    data_path = Path(audio_dir)
    data = list(data_path.glob("*.wav"))

    for file in data:
        process_audio(str(file), output_dir, yamnet_model)


def main(args):
    logging.info("Starting the main function...")

    audio_dir = args.audio_dir
    output_dir = args.output_dir

    process_audio_wrapper(audio_dir, output_dir)

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
