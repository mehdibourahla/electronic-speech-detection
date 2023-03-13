import tensorflow as tf
from utility import get_audio_path, get_config, get_cts_folds
from models.lstm_model import get_model
from draft import get_draft_model
from sklearn.metrics import classification_report
from keras.utils import to_categorical
import tensorflow_hub as hub
import pandas as pd
import librosa
import numpy as np
import os
import pandas as pd

dataset = "cts_recording"

config = get_config()
path = config["dir"] + f"/features/{dataset}_yamnet_features_social_interaction.npy"
if os.path.exists(path):
    features = np.load(path, allow_pickle=True)
else:
    data = get_audio_path(dataset)

    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    features = []

    for file in data:
        audio_wav, _ = librosa.load(file, sr=16000)
        _, embeddings, _ = yamnet_model(audio_wav)

        slug = file.name.split("-")[0]
        if slug == "100" or slug == "101" or slug == "111" or slug == "110":
            is_interaction = 1
        else:
            is_interaction = 0

        if slug == "010" or slug == "011" or slug == "110" or slug == "111":
            is_tv = 1
        else:
            is_tv = 0

        embeddings_num = tf.shape(embeddings)[0]
        is_interactions = tf.repeat(is_interaction, embeddings_num)
        is_tvs = tf.repeat(is_tv, embeddings_num)

        features.append((embeddings, is_tvs, is_interactions))

    np.save(path, pd.DataFrame(features))

X = np.array([point[0] for point in features], dtype="float32")
y_tv = np.array([point[1][0] for point in features], dtype="float32")
y_interaction = np.array([point[2][0] for point in features], dtype="float32")

y_tv = to_categorical(y_tv)
y_interaction = to_categorical(y_interaction)

tv_scores = []
model1_scores = []
model2_scores = []
for fold in get_cts_folds():
    X_train = np.delete(X, fold, axis=0)
    y_tv_train = np.delete(y_tv, fold, axis=0)
    y_interaction_train = np.delete(y_interaction, fold, axis=0)

    # Get the validation data
    X_val = X[fold]
    y_tv_val = y_tv[fold][:, 0]
    y_interaction_val = y_interaction[fold][:, 0]

    model_v1 = get_model(X[0].shape)
    model_v2 = get_draft_model(X[0].shape)

    model_v1.fit(X_train, y_interaction_train, epochs=10, batch_size=32, verbose=1)
    model_v2.fit(
        X_train, [y_tv_train, y_interaction_train], epochs=10, batch_size=32, verbose=1
    )

    y_pred_interaction_v1 = model_v1.predict(X_val)
    y_pred_tv, y_pred_interaction_v2 = model_v2.predict(X_val)

    y_pred_tv = [(tmp[0] > 0.5).astype(int) for tmp in y_pred_tv]
    y_pred_interaction_v1 = [
        (tmp[0] > 0.5).astype(int) for tmp in y_pred_interaction_v1
    ]
    y_pred_interaction_v2 = [
        (tmp[0] > 0.5).astype(int) for tmp in y_pred_interaction_v2
    ]

    report_tv = classification_report(y_tv_val, y_pred_tv, output_dict=True)
    report_interaction_v1 = classification_report(
        y_interaction_val, y_pred_interaction_v1, output_dict=True
    )
    report_interaction_v2 = classification_report(
        y_interaction_val, y_pred_interaction_v2, output_dict=True
    )
    tv_scores.append(report_tv)
    model1_scores.append(report_interaction_v1)
    model2_scores.append(report_interaction_v2)

tv_evaluation = {
    "accuracy": np.mean([score["accuracy"] for score in tv_scores]),
    "precision": np.mean([score["1.0"]["precision"] for score in tv_scores]),
    "recall": np.mean([score["1.0"]["recall"] for score in tv_scores]),
    "f1-score": np.mean([score["1.0"]["f1-score"] for score in tv_scores]),
}
model1_evaluation = {
    "accuracy": np.mean([score["accuracy"] for score in model1_scores]),
    "precision": np.mean([score["1.0"]["precision"] for score in model1_scores]),
    "recall": np.mean([score["1.0"]["recall"] for score in model1_scores]),
    "f1-score": np.mean([score["1.0"]["f1-score"] for score in model1_scores]),
}
model2_evaluation = {
    "accuracy": np.mean([score["accuracy"] for score in model2_scores]),
    "precision": np.mean([score["1.0"]["precision"] for score in model2_scores]),
    "recall": np.mean([score["1.0"]["recall"] for score in model2_scores]),
    "f1-score": np.mean([score["1.0"]["f1-score"] for score in model2_scores]),
}
