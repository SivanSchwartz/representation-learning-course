import argparse

import config
from model import NN, AutoEncoder, EncoderClassifier

# pretrain: AutoEncoder. pretrain_supervised and end_to_end: NN.
# linear_probe and finetune: EncoderClassifier, which loads the autoencoder's encoder.
MODES = ("pretrain", "pretrain_supervised", "linear_probe", "finetune", "end_to_end")


def parse_args():
    parser = argparse.ArgumentParser(description="Train one of the lecture models.")
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=MODES,
        default=config.MODES,
        help=(
            "Which run to train. "
            "pretrain: autoencoder on MNIST. "
            "pretrain_supervised: ResNet classifier on MNIST. "
            "linear_probe: frozen encoder plus a new head on Fashion-MNIST. "
            "finetune: encoder plus a new head, all weights, on Fashion-MNIST. "
            "end_to_end: ResNet from scratch on Fashion-MNIST. "
            "Pass several names to run them in order. Default: config.MODES."
        ),
    )
    return parser.parse_args()


def build_model(mode):
    if mode == "pretrain":
        return AutoEncoder(learning_rate=config.LEARNING_RATE, latent_dim=config.LATENT_DIM)

    if mode in ("linear_probe", "finetune"):
        return EncoderClassifier(
            learning_rate=config.FINETUNE_LEARNING_RATE if mode == "finetune" else config.LEARNING_RATE,
            num_classes=config.NUM_CLASSES,
            latent_dim=config.LATENT_DIM,
            encoder_path=config.ENCODER_PATH,
            freeze_encoder=(mode == "linear_probe"),
        )

    return NN(learning_rate=config.LEARNING_RATE, num_classes=config.NUM_CLASSES)
