import os
import math
import numpy as np
import librosa
import scipy.stats as stats
from scipy.signal import find_peaks
from utility import get_audio_path, get_config


def LinearityDegreeFeatures(power_normal):
    # Calculate signal power linearity degree features
    # input: power_normal
    # output: FV_LDF

    # Normalize power_vec as power_normal:
    # power_normal = power_vec / np.sum(power_vec)
    # From power_normal, calculate cumulative distribution of spectral power power_cdf:
    power_cdf = np.cumsum(power_normal)
    # Compute the correlation coefficients of power_cdf and store the results as rho:
    pearson_co = stats.pearsonr(power_cdf, np.arange(power_cdf.size))
    rho = pearson_co[0]
    # Compute the quadratic coefficients of power_cdf and store the results as q:
    x_values = np.arange(0, 8 + 7 / (power_cdf.size - 1), 8 / (power_cdf.size - 1))
    parameter_2 = np.polyfit(power_cdf, x_values, 2)
    q = parameter_2[0]
    # Form rho and q as FV_LDF:
    FV_LDF = np.array([rho, q])

    # Return power_cdf for plotting figures:
    return power_cdf, FV_LDF


def HighPowerFrequencyFeatures(FV_LFP, omega):
    # Calculate high power frequency
    # input: FV_LFP, omega
    # output: FV_HPF

    # 1. Find peaks from FV_LFP (returns the indices of found peaks):
    peaks_idx, _ = find_peaks(FV_LFP, height=0)
    # Obtain corresponding values of the peaks:
    peaks_val = FV_LFP[peaks_idx]
    # 2. Compute the threshold of selecting peaks using omega:
    T_peak = omega * max(peaks_val)
    # 3. Remove peaks lower than T_peak (insignificant peaks):
    peaks_idx = peaks_idx[np.where(peaks_val >= T_peak)]
    peaks_val = FV_LFP[peaks_idx]
    # 4. Obtain the number of remaining peaks:
    N_peak = peaks_idx.size
    # 5. Compute the mean of the locations of remaining peaks:
    mu_peak = peaks_idx.mean()
    # 6. Compute the standard deviation of the locations of remaining peaks:
    sigma_peak = np.std(peaks_idx)
    # 7. Use a 6-order polynomial to fit FV_LFP and take first 32 estimatied values as P_est:
    parameter_6 = np.polyfit(np.arange(FV_LFP.size), FV_LFP, 6)
    value_est = np.polyval(parameter_6, np.arange(FV_LFP.size))

    P_est = value_est[0:32]
    # Construct FV_HPF (insert N_peak, mu_peak and sigma_peak in fornt of P_est):
    FV_HPF = np.insert(P_est, 0, [N_peak, mu_peak, sigma_peak])
    return FV_HPF


def lpc_to_lpcc(lpc):
    # Based on given LPC, calculate LPCC:
    lpcc = []
    order = lpc.size - 1
    # The 1st element equals ln(order):
    lpcc.append(math.log(order))
    lpcc.append(lpc[1])
    for i in range(2, order + 1):
        sum_1 = 0
        for j in range(1, i):
            sum_1 += j / i * lpc[i - j - 1] * lpcc[j]
        c = -lpc[i - 1] + sum_1
        lpcc.append(c)
    return lpcc[1:13]


def extract_lpcc(wav_path, order):
    y, _ = librosa.load(wav_path, sr=16000)
    lpc = librosa.lpc(y, order=order)
    lpcc = np.array(lpc_to_lpcc(lpc))
    return lpcc


def _stft(y):
    n_fft, hop_length, _ = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)


def _stft_parameters():
    # n_fft = (num_freq - 1) * 2
    n_fft = 2048
    hop_length = 128
    # hop_length = int(frame_shift_ms / 1000 * sample_rate)
    # win_length = int(frame_length_ms / 1000 * sample_rate)
    win_length = 512
    return n_fft, hop_length, win_length


def load_void_features(dataset):
    config = get_config()
    path = config["dir"] + f"/features/{dataset}_void_features.npy"
    if os.path.exists(path):
        features = np.load(path, allow_pickle=True)
    else:
        audioset = get_audio_path(dataset)

        # Initialize parameters:
        W = 14
        # Peak selection threshold:
        omega = 0.3
        # Number of FFT points:
        nfft = 2048
        # Calculate the number of segments k in S_pow:
        k = int((nfft / 2 + 1) / W)

        # Create an empty Numpy array to store extracted features as well as corresponding labels:
        features = np.array(np.zeros((len(audioset), 98)), dtype=object)

        for idx, file in enumerate(audioset):

            # ------ Stage 1: Signal transformation ------
            # Read the input signal:
            signal, _ = librosa.load(file, sr=16000)

            # Compute STFT for the input signal:
            sig_stft = _stft(signal)

            # Compute S_pow from STFT:
            S_pow = np.sum(np.abs(sig_stft) ** 2 / nfft, axis=1)

            # ------ Stage 2: Feature Extraction ------
            # Calculate the sum of power in each segment (in total k segments):
            power_vec = np.zeros(k)
            for i in np.arange(k):
                power_vec[i] = np.sum(S_pow[i * W : (i + 1) * W])
            # Normalize power_vec as power_normal:
            power_normal = power_vec / np.sum(power_vec)

            # Feature 1: FV_LFP - low frequencies power features
            FV_LFP = power_normal[0:48] * 100

            # Feature 2: FV_LDF - signal power linearity degree features
            _, FV_LDF = LinearityDegreeFeatures(power_normal)

            # Feature 3: FV_HPF - high power frequency features
            FV_HPF = HighPowerFrequencyFeatures(FV_LFP, omega)

            # Feature 4: FV_LPC - linear prediction cesptrum coefficients
            FV_LPC = extract_lpcc(file, 12)

            # Construct the final feature of length 97 (= 2 + 35 + 12 + 48):
            FV_Void = np.concatenate((FV_LDF, FV_HPF, FV_LPC, FV_LFP))

            label = file.name.split("-")[0]
            if label == "010" or label == "011" or label == "111" or label == "110":
                label = 1
            else:
                label = 0
            features[idx, 0:97] = FV_Void
            features[idx, 97] = label
        np.save(config["dir"] + f"/features/{dataset}_void_features.npy", features)
    X, y = np.split(features, indices_or_sections=(97,), axis=1)

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32").ravel()

    return X, y
