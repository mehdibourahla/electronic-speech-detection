import os
import numpy as np
import librosa
import logging
import argparse
import numpy as np
import scipy
import soundfile as sf
import librosa
from scipy.stats import zscore
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Configure logging
logging.basicConfig(
    filename="features.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def calculate_energy_metric(audio_path):
    audio_data, sample_rate = sf.read(audio_path)

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Define necessary parameters
    window_size = 0.1  # in seconds
    fft_size = 4096
    low_frequency = 20  # in Hz
    high_frequency = 250  # in Hz
    cutoff_frequency = 80  # in Hz

    # Step 1: Sliding window
    window_length_samples = int(window_size * sample_rate)
    windows = librosa.util.frame(
        audio_data, frame_length=window_length_samples, hop_length=window_length_samples
    )

    # Step 2: FFT and cropping
    low_index = int(low_frequency * fft_size / sample_rate)
    high_index = int(high_frequency * fft_size / sample_rate)
    cutoff_index = int(cutoff_frequency * fft_size / sample_rate)

    fft_frames = np.empty((windows.shape[1], high_index - low_index + 1))

    for i, window in enumerate(windows.T):
        fft = abs(scipy.fftpack.fft(window, n=fft_size))
        cropped_fft = fft[low_index : high_index + 1]
        fft_frames[i] = cropped_fft

    energy_balance_metric = []

    # Step 3: Integration and normalization
    for cropped_fft in fft_frames:
        spectral_energy = np.sum(cropped_fft)
        normalized_energy_curve = cropped_fft / spectral_energy

        # Step 4: Energy balance metric computation
        sub_bass_energy = np.sum(normalized_energy_curve[:cutoff_index])
        total_energy = np.sum(normalized_energy_curve)
        energy_balance_metric.append(sub_bass_energy / total_energy)

    energy_balance_metric = np.array(energy_balance_metric)

    # Step 5: Outliers removal based on skewness and median selection
    z_scores = np.abs(zscore(energy_balance_metric))

    # Remove outliers (data points with Z-score > 3)
    energy_balance_metric = energy_balance_metric[z_scores < 3]

    final_energy_balance = np.median(energy_balance_metric)
    final_energy_balance = statistics.median(energy_balance_metric)

    return final_energy_balance


def process_audio(path, output_dir):
    logging.info(f"Processing audio: {path}")
    median_energy_balance = calculate_energy_metric(path)
    filename = path.split("/")[-1]

    metric_file = f"{output_dir+'/'+filename}.npy"
    try:
        np.save(metric_file, median_energy_balance)
    except Exception as e:
        logging.error(f"Error saving embeddings for {filename}: {e}")
        return None


def process_audio_wrapper(audio_dir, output_dir):
    data_path = Path(audio_dir)
    data = list(data_path.glob("*.wav"))
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_audio, str(file), output_dir) for file in data
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
