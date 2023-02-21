import tensorflow as tf
from utility import get_audio_path, get_config
import tensorflow_hub as hub
import pandas as pd
import librosa
import numpy as np
import os
import pandas as pd


def load_yamnet_features(dataset):

    config = get_config()
    path = config["dir"] + f"/features/{dataset}_yamnet_features.npy"
    if os.path.exists(path):
        features = np.load(path, allow_pickle=True)
    else:
        data = get_audio_path(dataset)

        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

        features = []

        for file in data:
            audio_wav, _ = librosa.load(file, sr=16000)
            _, embeddings, _ = yamnet_model(audio_wav)

            label = file.name.split("-")[0]
            if label == "010" or label == "011" or label == "111" or label == "110":
                label = 1
            else:
                label = 0

            embeddings_num = tf.shape(embeddings)[0]
            labels = tf.repeat(label, embeddings_num)

            features.append((embeddings, labels))

        np.save(path, pd.DataFrame(features))

    X = np.array([point[0] for point in features], dtype="float32")
    y = np.array([point[1][0] for point in features], dtype="float32")
    return X, y
