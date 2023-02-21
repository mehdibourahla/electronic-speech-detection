import os
import librosa
from tqdm import tqdm
import numpy as np
from scipy.fftpack import fft

def calculate_energy_metric(audioset):
    f_min = 20
    f_max = 250
    n = 4096

    median = []
    for file in tqdm(audioset):
        audio, fs = librosa.load(file, sr=None)
        # Applying a sliding window of 0.1s
        window_size = int(fs * 0.1)
        sliding_windows = np.split(
            audio, range(window_size, audio.shape[0], window_size)
        )

        audio_energy = []

        for window in sliding_windows:
            # Extract the FFT from each window and get Magnitude in dB
            fft_magnitude = np.abs(fft(window, n))

            # Crop the FFT to only include the frequencies of interest
            fft_magnitude = fft_magnitude[f_min:f_max]

            # Convert Magnitude to dB
            with np.errstate(divide="raise"):
                try:
                    mag_freq = 20 * np.log10(fft_magnitude)
                except:
                    print("Something went wrong in Log10")

            # Integrate over mag_freq to compute the spectral energy and normalize
            psd = mag_freq**2
            with np.errstate(invalid="raise"):
                try:
                    psd = psd / np.max(psd)
                except:
                    print("Something went wrong in Division")
            energy_metric = np.mean(psd[60:70])

            audio_energy.append(energy_metric)

        median.append(np.median(audio_energy))

    return median
