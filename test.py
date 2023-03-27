import numpy as np
from yamnet import load_yamnet_features
from keras.utils import to_categorical
from models.lstm_model import get_model
from sklearn.metrics import classification_report


### YAMNET + LSTM
X_test, y_test = load_yamnet_features("my_recording")
X_plus, y_plus = load_yamnet_features("kelly_recording_closed")
X_test = np.append(X_test, X_plus, axis=0)
y_test = np.append(y_test, y_plus)

X_plus, y_plus = load_yamnet_features("kelly_recording_open")
X_test = np.append(X_test, X_plus, axis=0)
y_test = np.append(y_test, y_plus)

X, y = load_yamnet_features("cts_recording")

X = np.mean(X.reshape((X.shape[0], 31, 2, 1024)), axis=2)
y = to_categorical(y)

X_test = np.mean(X_test.reshape((X_test.shape[0], 31, 2, 1024)), axis=2)

model = get_model(X[0].shape)
model.fit(X, y, epochs=10, batch_size=32, verbose=1)
y_pred = model.predict(X_test)

y_pred = [(tmp[0] < 0.5).astype(int) for tmp in y_pred]
y_test = [int(i) for i in y_test]
report = classification_report(y_test, y_pred, output_dict=True)
