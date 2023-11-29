import os
import numpy as np
import librosa
import argparse
import scipy
import soundfile as sf
from scipy.stats import zscore
import statistics
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import resampy


def calculate_energy_metric(audio_data, sample_rate):
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
        spectral_energy = np.sum(cropped_fft**2)
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


def process_audio(audio_dir, output_dir, duration=5, sample_rate=16000):
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.wav"))

    # Open a text file to store the results
    with open(output_dir + "/ebm_results_v5.txt", "a") as results_file:
        for audio_file in tqdm(audio_files):
            try:
                wav_data, sr = sf.read(audio_file, dtype=np.int16)
                if len(wav_data.shape) > 1:
                    wav_data = np.mean(wav_data, axis=1)

                if sr != sample_rate:
                    wav_data = resampy.resample(wav_data, sr, sample_rate)

                pace = duration * sample_rate

                num_full_chunks = (
                    len(wav_data) // pace
                )  # Compute the number of full chunks

                for i in range(num_full_chunks):  # Loop only over full chunks
                    start_time = i * pace
                    end_time = start_time + pace
                    chunk = wav_data[start_time:end_time]

                    # Compute the energy balance metric for the chunk
                    ebm = calculate_energy_metric(chunk, sample_rate)

                    # Write the filename and the ebm value to the text file
                    results_file.write(f"{audio_file.stem}_{i}: {ebm}\n")

            except Exception as e:
                print(f"Error processing audio file {audio_file}: {e}")
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

    process_audio(args.data_dir, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
