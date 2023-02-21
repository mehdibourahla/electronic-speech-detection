import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from utility import get_cts_folds, export_model
from models.lstm_model import get_model
from yamnet import load_yamnet_features
from void import load_void_features
from yamnet_pretrained import load_yamnet_inference

### YAMNET + LSTM
X, y = load_yamnet_features("cts_recording")

yamnet_scores = []
for fold in get_cts_folds():
    X_train = np.delete(X, fold, axis=0)
    y_train = np.delete(y, fold)

    # Get the validation data
    X_val = X[fold]
    y_val = y[fold]

    model = get_model(X[0].shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    y_pred = model.predict(X_val)
    y_pred = (y_pred > 0.5).astype(int)
    report = classification_report(y_val, y_pred, output_dict=True)
    yamnet_scores.append(report)

yamnet_accuracy = np.mean([score["accuracy"] for score in yamnet_scores])
yamnet_precision = np.mean([score["1.0"]["precision"] for score in yamnet_scores])
yamnet_f1 = np.mean([score["1.0"]["f1-score"] for score in yamnet_scores])
yamnet_recall = np.mean([score["1.0"]["recall"] for score in yamnet_scores])

### Export YAMNET + LSTM
model = get_model(X[0].shape)
model.fit(X, y, epochs=10, batch_size=32, verbose=1)
export_model(model, 'YAMNet_LSTM')

### VOID + SVM
void_scores = []
X, y = load_void_features("cts_recording")
for fold in get_cts_folds():
    X_train = np.delete(X, fold, axis=0)
    y_train = np.delete(y, fold)

    # Get the validation data
    X_val = X[fold]
    y_val = y[fold]

    classifier = SVC(kernel="rbf", gamma="auto", C=0.5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    void_scores.append(report)

void_accuracy = np.mean([score["accuracy"] for score in void_scores])
void_precision = np.mean([score["1.0"]["precision"] for score in void_scores])
void_f1 = np.mean([score["1.0"]["f1-score"] for score in void_scores])
void_recall = np.mean([score["1.0"]["recall"] for score in void_scores])


### YAMNet Inference
yamnet_results = load_yamnet_inference("cts_recording")
grouped_df = yamnet_results.groupby("Labels")["isTV"].median()
closest = []
for x in yamnet_results["isTV"]:
    closest.append(min(grouped_df[0], grouped_df[1], key=lambda num: abs(num - x)))

y_pred = list((closest == grouped_df[1]).astype(int))
y_true = list(yamnet_results["Labels"])
report = classification_report(y_true, y_pred, output_dict=True)
### Visualize results

# Plot bar chart of mean and standard deviation of precision, recall, and F1 score
labels = ["Accuracy", "Precision", "F1 Score", "Recall"]
x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(
    x - width / 2,
    [yamnet_accuracy, yamnet_precision, yamnet_f1, yamnet_recall],
    width,
    label="YAMNet + LSTM",
)
rects2 = ax.bar(
    x + width / 2,
    [void_accuracy, void_precision, void_f1, void_recall],
    width,
    label="Void + SVM",
)

rects3 = ax.bar(
    x + 3*width/2,
    [
        report["accuracy"],
        report["1"]["precision"],
        report["1"]["f1-score"],
        report["1"]["recall"],
    ],
    width,
    label="Pretrained YAMNet",
)

ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
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

plt.show()
