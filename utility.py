import tensorflow as tf
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score


def get_audio_path(source):
    with open("config.json") as f:
        config = json.load(f)
    data_path = Path(config[source])
    return list(data_path.glob("*.wav"))


def get_data(data):
    X, y = np.split(data, indices_or_sections=(97,), axis=1)

    X = list(X)
    y = list(y)

    X = [list(x) for x in X]
    y = [list(x)[0] for x in y]

    return X, y


def get_config():
    with open("config.json") as f:
        config = json.load(f)
    return config


def get_cts_folds():
    a_elements = []
    for i in range(0, 540, 80):
        for j in range(0, 20):
            a_elements.append(i + j)
    b_elements = []
    for i in range(20, 540, 80):
        for j in range(0, 20):
            b_elements.append(i + j)
    c_elements = []
    for i in range(40, 540, 80):
        for j in range(0, 20):
            c_elements.append(i + j)
    d_elements = []
    for i in range(60, 540, 80):
        for j in range(0, 20):
            d_elements.append(i + j)
    return a_elements, b_elements, c_elements, d_elements


def export_model(model, name):
    config = get_config()
    path = config["dir"] + f"/exports/{name}.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    with tf.io.gfile.GFile(path, "wb") as f:
        f.write(tflite_model)
