from transformer import TransformerClassifier
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
    filename="transformer.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    logging.info(f"Found {len(physical_devices)} GPUs: {physical_devices}")
else:
    logging.warning("No GPUs found")


def transformer_model(input_dim):
    # Define model parameters
    num_layers = 2
    embed_dim = input_dim[1]  # 1024
    num_heads = 2
    ff_dim = 512
    num_classes = 2
    input_shape = (None, embed_dim)

    # Initialize and compile model
    model = TransformerClassifier(
        num_layers, embed_dim, num_heads, ff_dim, input_shape, num_classes
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def main(args):
    logging.info("Starting the Transformer model...")
    args.model_name = "transformer"
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
    model = train_model(transformer_model, training_generator, validation_generator)
    results = evaluate_model(model, test_generator)
    save_training_results(results, args)
    export_model(transformer_model, full_data_generator, args)

    logging.info("Done with Transformer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
