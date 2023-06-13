import os
import argparse
import keras_tuner as kt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model_builder import CNNHyperModel
from model import load_balanced_data, load_ground_truth
from data_generator import DataGenerator, DataGeneratorMultiple


def get_early_stopping_callback(patience=10):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=0,
        mode="auto",
        restore_best_weights=True,
    )


def get_hyper_model(directory, train_gen, val_gen, model_builder, epochs):
    tuner = kt.Hyperband(
        model_builder,
        objective="val_loss",
        factor=3,
        directory=directory,
    )

    tuner.search(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        callbacks=[get_early_stopping_callback(patience=3)],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Hyperparameter search complete!")

    return tuner.hypermodel.build(best_hps)


def train(params, model_builder, train_gen, val_gen):
    print("GPU Available: ", tf.config.list_physical_devices("GPU"))

    output_dir = params.output_dir
    epochs = params.epochs

    model = get_hyper_model(
        directory=output_dir,
        train_gen=train_gen,
        val_gen=val_gen,
        model_builder=model_builder,
        epochs=epochs,
    )

    checkpoint_dir = f"{output_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = (
        checkpoint_path
        + "/{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{accuracy:.4f}-{val_accuracy:.4f}.hdf5"
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)

    early_stopping_callback = get_early_stopping_callback(patience=10)

    print(len(train_gen))
    # Train model for n epochs
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )


def initialize_args(parser):
    # Input paths

    # Add argument for number of epochs
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the directory containing NPY files",
    )
    parser.add_argument(
        "--gt_dir", required=True, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--data_dir2",
        default=None,
        help="Path to the directory containing NPY files",
    )
    parser.add_argument(
        "--gt_dir2", default=None, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    # Define all your command line arguments here
    args = parser.parse_args()

    # Load data
    if args.data_dir2 and args.gt_dir2:
        print("Loading data from two directories")
        balanced_data, dir_mapping = load_balanced_data(
            args.gt_dir, args.gt_dir2, args.data_dir, args.data_dir2
        )
        generator_class = DataGeneratorMultiple
        generator_args = (dir_mapping,)
    else:
        balanced_data = load_ground_truth(args.gt_dir)
        generator_class = DataGenerator
        generator_args = (args.data_dir,)
    balanced_data_train, balanced_data_val = train_test_split(
        balanced_data, test_size=0.2, random_state=42
    )
    training_generator = generator_class(*generator_args, balanced_data_train)
    validation_generator = generator_class(*generator_args, balanced_data_val)

    # Get a batch of data
    data_batch, _ = training_generator.__getitem__(0)

    # Get the shape of a single sample
    input_shape = data_batch[0].shape
    model_builder = CNNHyperModel(input_shape)
    train(args, model_builder, training_generator, validation_generator)
