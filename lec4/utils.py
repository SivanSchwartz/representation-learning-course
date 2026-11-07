import argparse

import config
from model import NN, AutoEncoder, EncoderClassifier, SupervisedEncoder, SimCLR

# pretrain: AutoEncoder.
# pretrain_supervised: ResNet-18 NN. pretrain_supervise_encoder: SupervisedEncoder.
# pretrain_simclr: SimCLR. The loss is config.CONTRASTIVE_LOSS, a function in losses.py.
# linear_probe and finetune: EncoderClassifier, which loads a saved encoder.
# end_to_end: ResNet-18 NN on the target dataset.
MODES = (
    "pretrain",
    "pretrain_supervised",
    "pretrain_supervise_encoder",
    "pretrain_simclr",
    "linear_probe",
    "finetune",
    "end_to_end",
)


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
            "pretrain_supervise_encoder: small encoder plus a head, from scratch, on MNIST. "
            "pretrain_simclr: contrastive pretraining on MNIST. "
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

    if mode == "pretrain_simclr":
        return SimCLR(
            learning_rate=config.LEARNING_RATE,
            latent_dim=config.LATENT_DIM,
            projector_dim=config.PROJECTOR_DIM,
            temperature=config.TEMPERATURE,
            margin=config.TRIPLET_MARGIN,
            loss_name=config.CONTRASTIVE_LOSS,
        )

    if mode == "pretrain_supervise_encoder":
        return SupervisedEncoder(
            learning_rate=config.LEARNING_RATE,
            num_classes=config.NUM_CLASSES,
            latent_dim=config.LATENT_DIM,
        )

    if mode in ("linear_probe", "finetune"):
        return EncoderClassifier(
            learning_rate=config.FINETUNE_LEARNING_RATE if mode == "finetune" else config.LEARNING_RATE,
            num_classes=config.NUM_CLASSES,
            latent_dim=config.LATENT_DIM,
            encoder_path=config.ENCODER_PATH,
            freeze_encoder=(mode == "linear_probe"),
        )

    return NN(learning_rate=config.LEARNING_RATE, num_classes=config.NUM_CLASSES)
