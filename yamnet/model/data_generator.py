import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, balanced_data, batch_size=32, dim=(32, 32, 32), shuffle=True):
        "Initialization"
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = balanced_data.index.tolist()
        self.labels = balanced_data.tv.values
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return (len(self.list_IDs) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self._data_generation(list_IDs_temp)

        return X, y

    @property
    def classes(self):
        return to_categorical(self.labels)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"
        X = []
        y = []

        # Generate data
        for ID in list_IDs_temp:
            # Load sample and append to list
            try:
                sequence = np.load(ID, allow_pickle=True)
                X.append(sequence)

                # Store class
                y.append(self.labels[self.list_IDs.index(ID)])
            except Exception as e:
                continue

        X = pad_sequences(
            X, maxlen=31, dtype="float32", padding="post", truncating="post"
        )
        y = np.array(y)
        y = to_categorical(y)

        return X, y

    def load_all_data(self):
        return self._data_generation(self.list_IDs)
