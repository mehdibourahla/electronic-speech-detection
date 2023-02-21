import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from utility import get_audio_path, get_config
import os
import librosa
import pandas as pd


def load_yamnet_inference(dataset):

    config = get_config()
    path = config["dir"] + f"/features/{dataset}_yamnet_inference.npy"
    if os.path.exists(path):
        features = np.load(path, allow_pickle=True)
    else:
        data = get_audio_path(dataset)

        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
        class_names = list(pd.read_csv(class_map_path)["display_name"])

        features = []
        for file in data:
            label = file.name.split("-")[0]
            if label == "010" or label == "011" or label == "111" or label == "110":
                label = 1
            else:
                label = 0
            with open(file, "rb") as f:
                audio_wav, _ = librosa.load(f, sr=16000)
            scores, _, _ = yamnet_model(audio_wav)
            class_scores = tf.reduce_mean(scores, axis=0)
            _, top_indices = tf.math.top_k(class_scores, k=10)

            inferred_classes = [class_names[idx] for idx in top_indices]
            tv_score, radio_score = [
                class_scores[class_names.index(name)].numpy()
                for name in ["Television", "Radio"]
            ]
            features.append([label, inferred_classes, tv_score, radio_score])
        features = np.array(features)
        np.save(config["dir"] + f"/features/{dataset}_yamnet_inference.npy", features)

    y_pred = features[:, 2] + features[:, 3]
    df = pd.DataFrame(columns=["Labels", "isTV"])
    df["Labels"] = features[:, 0]
    df["isTV"] = y_pred
    return df
