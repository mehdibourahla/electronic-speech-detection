from tensorflow.keras import layers, optimizers, Input, Model
import logging
import tensorflow as tf
import argparse
import os
from data_generator import DataGenerator
from model import (
    initialize_args,
    load_ground_truth,
    train_model,
    save_training_results,
    export_model,
    split_data,
    evaluate_model,
)

os.environ["TFHUB_CACHE_DIR"] = "/users/mbourahl/.cache"

# Configure logging
logging.basicConfig(
    filename="lstm.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    logging.info(f"Found {len(physical_devices)} GPUs: {physical_devices}")
else:
    logging.warning("No GPUs found")


# LSTM Model
def lstm_model(input_shape):
    logging.info(f"Creating model with input shape {input_shape}")
    adam = optimizers.Adam(3e-4)
    inputs = Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(adam, "categorical_crossentropy", metrics=["accuracy"])

    return model


def main(args):
    logging.info("Starting the LSTM model...")
    args.model_name = "lstm"
    balanced_data = load_ground_truth(args.gt_dir)

    balanced_data_train, balanced_data_val, balanced_data_test = split_data(
        balanced_data
    )

    # Datasets
    training_generator = DataGenerator(args.data_dir, balanced_data_train)
    validation_generator = DataGenerator(args.data_dir, balanced_data_val)
    test_generator = DataGenerator(args.data_dir, balanced_data_test)
    full_data_generator = DataGenerator(
        args.data_dir,
        balanced_data,
    )
    model = train_model(lstm_model, training_generator, validation_generator)
    results = evaluate_model(model, test_generator)
    save_training_results(results, args)
    export_model(lstm_model, full_data_generator, args)

    logging.info("Done with LSTM.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
