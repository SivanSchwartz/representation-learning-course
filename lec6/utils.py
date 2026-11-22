import argparse

import config
from model import NN, AutoEncoder, EncoderClassifier, SupervisedEncoder, SimCLR, MoCo, BYOL

# One name from this tuple. --mode is required; config does not pick one.
# pretrain: AutoEncoder.
# pretrain_supervised: ResNet-18 NN. pretrain_supervise_encoder: SupervisedEncoder.
# pretrain_simclr: SimCLR. The loss is config.CONTRASTIVE_LOSS, a function in losses.py.
# pretrain_moco: MoCo. Negatives come from a queue, not from the rest of the batch.
# pretrain_byol: BYOL. A predictor matches a momentum target. There are no negatives.
# linear_probe and finetune: EncoderClassifier, which loads a saved encoder.
# end_to_end: ResNet-18 NN.
MODES = (
    "pretrain",
    "pretrain_supervised",
    "pretrain_supervise_encoder",
    "pretrain_simclr",
    "pretrain_moco",
    "pretrain_byol",
    "linear_probe",
    "finetune",
    "end_to_end",
)
DATASETS = {"mnist": "MNIST", "fashionmnist": "FashionMNIST"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train one lecture model.")
    parser.add_argument(
        "--mode",
        choices=MODES,
        required=True,
        help="Which model to train.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        required=True,
        help="mnist or fashionmnist.",
    )
    args = parser.parse_args()
    config.DATASET = DATASETS[args.dataset]
    config.ENCODER_PATH = f"{args.dataset}_encoder.pt"
    return args


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

    if mode == "pretrain_moco":
        return MoCo(
            learning_rate=config.LEARNING_RATE,
            latent_dim=config.LATENT_DIM,
            projector_dim=config.PROJECTOR_DIM,
            K=config.MOCO_K,
            m=config.MOCO_M,
            T=config.TEMPERATURE,
        )

    if mode == "pretrain_byol":
        return BYOL(
            learning_rate=config.LEARNING_RATE,
            latent_dim=config.LATENT_DIM,
            projector_dim=config.PROJECTOR_DIM,
            m_base=config.BYOL_M,
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
