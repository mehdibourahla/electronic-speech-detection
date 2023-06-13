import os
import argparse
import keras_tuner as kt
import tensorflow as tf

from model_builder import CNNHyperModel, LSTMHyperModel, TransformerHyperModel
from model import load_balanced_data, load_ground_truth, evaluate_model
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


def get_hyper_model(
    directory, project_name, train_gen, val_gen, model_builder, hp_max_epochs
):
    tuner = kt.Hyperband(
        model_builder,
        objective="val_loss",
        max_epochs=hp_max_epochs,
        project_name=project_name,
        factor=3,
        directory=directory,
    )

    tuner.search(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_gen),
        epochs=hp_max_epochs,
        callbacks=[get_early_stopping_callback(patience=3)],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Hyperparameter search complete!")

    return tuner.hypermodel.build(best_hps)


def train(params, model_builder, train_gen, val_gen):
    print("GPU Available: ", tf.config.list_physical_devices("GPU"))

    output_dir = params.output_dir
    epochs = params.epochs
    hp_max_epochs = params.hp_max_epochs
    project_name = params.project_name

    model = get_hyper_model(
        directory=output_dir,
        project_name=project_name,
        train_gen=train_gen,
        val_gen=val_gen,
        model_builder=model_builder,
        hp_max_epochs=hp_max_epochs,
        epochs=epochs,
    )

    checkpoint_dir = f"{output_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = (
        checkpoint_dir
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
    parser.add_argument(
        "--i_fold", type=int, help="Split number (which fold is the test set?)"
    )
    parser.add_argument("--j_subfold", type=int, help="Validation fold")
    parser.add_argument(
        "--model_type", type=str, default="Transformer", help="LSTM, CNN or Transformer"
    )
    parser.add_argument(
        "--hp_max_epochs", type=int, default=20, help="max_epochs in Hyperband opt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    # Define all your command line arguments here
    args = parser.parse_args()
    i_fold = args.i_fold
    j_subfold = args.j_subfold
    model_type = args.model_type.lower()

    args.project_name = f"{model_type}_fold_{i_fold}_subfold_{j_subfold}"
    args.output_dir = f"{args.output_dir}/{model_type}"
    os.makedirs(args.output_dir, exist_ok=True)

    models_dict = {
        "cnn": CNNHyperModel,
        "lstm": LSTMHyperModel,
        "transformer": TransformerHyperModel,
    }

    model_func = models_dict[model_type]

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

    num_folds = 5
    folds = []
    n_records = len(balanced_data)
    records_per_fold = n_records // num_folds

    for i in range(num_folds):
        folds.append(
            balanced_data.iloc[
                i * records_per_fold : min((i + 1) * records_per_fold, n_records)
            ]
        )

    current_fold = folds[i_fold]
    remaining_folds = [
        fold_instance for fold_instance in folds if fold_instance != current_fold
    ]
    test_ids = current_fold

    holdout_fold = remaining_folds[j_subfold]
    train_ids = [
        id_instance
        for fold_instance in remaining_folds
        if fold_instance != holdout_fold
        for id_instance in fold_instance
    ]
    val_ids = holdout_fold

    # Create data generators
    train_data = balanced_data.iloc[train_ids]
    train_gen = generator_class(*generator_args, train_data)

    val_data = balanced_data.iloc[val_ids]
    val_gen = generator_class(*generator_args, val_data, batch_size=1)

    # Perform the model training and hyperparameter tuning
    # Get a batch of data
    data_batch, _ = train_gen.__getitem__(0)
    input_shape = data_batch[0].shape

    model = model_func(input_shape)
    train(args, model, train_gen, val_gen)

    # Evaluate the model on the test data
    test_data = balanced_data.iloc[test_ids]
    test_gen = generator_class(*generator_args, test_data, batch_size=1)

    output_path = f"{args.output_dir}/fold_{i_fold}_subfold_{j_subfold}"
    evaluate_model(model, test_gen, output_path)
