from utility import get_features_path, export_model
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from models.lstm_model import get_model
import matplotlib.pyplot as plt

def extract_label_from_path(path):
    filename = os.path.basename(path)
    label = filename[0]
    return label


def extract_features_and_labels(data_list):
    X = []
    y = []
    for data in data_list:
        for feature in data:
            embeddings = feature[:, :-1]
            labels = feature[:, -1]
            X.append(embeddings)
            y.append(labels[0])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")

def preprocess_data(X, y, batch_size=31):
    X = np.mean(X.reshape((-1, batch_size, 2, 1024)), axis=2)
    y = to_categorical(y, num_classes=2)
    return X, y

# Import features from different datasets
my_recording_path = get_features_path("my_recording")
common_voice_path = get_features_path("common_voice")

cts_data = np.load(get_features_path("cts_recording")[0], allow_pickle=True)
cts_X = np.array([point[0] for point in cts_data], dtype="float32")
cts_y = np.array([point[1][0] for point in cts_data], dtype="float32")

kelly_data = np.load(get_features_path("kelly_recording")[0], allow_pickle=True)
kelly_X = np.array([point[0] for point in kelly_data], dtype="float32")
kelly_y = np.array([point[1][0] for point in kelly_data], dtype="float32")

my_recording_data = [np.load(path, allow_pickle=True) for path in my_recording_path]
my_recording_X, my_recording_y = extract_features_and_labels(my_recording_data)

common_voice_data = [np.load(path, allow_pickle=True) for path in common_voice_path]
common_voice_preprocessed = []
for data in common_voice_data:
    for instance in data:
        common_voice_preprocessed.append(instance)
common_voice_preprocessed = np.array(common_voice_preprocessed, dtype="float32")
# Calculate the value of X
X = common_voice_preprocessed.shape[0] // 62

# Calculate the number of rows needed for padding
padding_rows = (X * 62 + 62) - common_voice_preprocessed.shape[0]

# Pad the array with zeros
padded_array = np.pad(
    common_voice_preprocessed, ((0, padding_rows), (0, 0)), mode="constant", constant_values=0
)

# Reshape the padded array
common_voice_preprocessed = padded_array.reshape(X + 1, 62, 1025)
common_voice_X = []
common_voice_y = []
for feature in common_voice_preprocessed:
    embeddings = feature[:, :-1]
    labels = feature[:, -1]
    common_voice_X.append(embeddings)
    common_voice_y.append(labels[0])
common_voice_X = np.array(common_voice_X, dtype="float32")
common_voice_y = np.array(common_voice_y, dtype="float32")

# Apply preprocessing to all datasets
common_voice_X, common_voice_y = preprocess_data(common_voice_X, common_voice_y)
my_recording_X, my_recording_y = preprocess_data(my_recording_X, my_recording_y)
kelly_X, kelly_y = preprocess_data(kelly_X, kelly_y)
cts_X, cts_y = preprocess_data(cts_X, cts_y)

# Data concatenation
X = np.concatenate([my_recording_X, common_voice_X, kelly_X, cts_X], axis=0)
y = np.concatenate([my_recording_y, common_voice_y, kelly_y, cts_y], axis=0)

# Parameters
num_folds = 4
input_shape = (31, 1024)
batch_size = 32
epochs = 10

classification_reports = []
# K-fold cross-validation
kfold = KFold(n_splits=num_folds, shuffle=True)
fold = 1
for train_idx, test_idx in kfold.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Create and train the model
    model = get_model(input_shape)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

     # Get binary predictions
    y_pred = (model.predict(X_test) > 0.5).astype('int32')

    # Calculate the classification report for the current fold
    report = classification_report(y_test, y_pred, output_dict=True)
    # Save the classification report to an array
    classification_reports.append(report)

datasets = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4']
precision = [d['1']['precision'] for d in classification_reports]
recall = [d['1']['recall'] for d in classification_reports]
f1_score = [d['1']['f1-score'] for d in classification_reports]

x = np.arange(len(datasets))
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-score')

ax.set_ylabel('Scores')
ax.set_title('Scores for Positive Class (1) across Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

fig.tight_layout()

# Add values as text annotations to the bars
for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2.0,
        height - height / 2,
        "{:.2f}".format(height),
        ha="center",
        va="bottom",
        color="white",
    )

# Show the bar chart
plt.show()

### Export Model
model = get_model(X[0].shape)
model.fit(X, y, epochs=10, batch_size=32, verbose=1)
export_model(model, "YAMNet_LSTM_v2")
