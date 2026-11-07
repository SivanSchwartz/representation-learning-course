# Training hyperparameters
NUM_CLASSES = 10  # MNIST and Fashion-MNIST both have 10 classes (but different meanings!)
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 100
LATENT_DIM = 64  # length of z. AutoEncoder and EncoderClassifier must use the same value.

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "auto"  # GPU on Colab, CPU on this Mac
DEVICES = 1
PRECISION = "16-mixed"  # same as 16 (the old name); falls back to bf16 on CPU

# Models (see model.py), picked per experiment in train.py:
#   "pretrain"                    : AutoEncoder - reconstruction on MNIST, saves the encoder
#   "pretrain_supervised"         : NN - ResNet-18 trained with cross-entropy on MNIST
#   "pretrain_supervise_encoder"  : SupervisedEncoder - small encoder + head from scratch on MNIST, saves the encoder
#   "pretrain_simclr"             : SimCLR - contrastive loss on two views of MNIST, saves the encoder (not the projector)
#   "linear_probe"                : EncoderClassifier - that encoder FROZEN + a new head, Fashion-MNIST
#   "finetune"                    : EncoderClassifier - that encoder + a new head, all weights, Fashion-MNIST
#   "end_to_end"                  : NN - ResNet-18 from scratch on Fashion-MNIST
# These train on the source dataset. The others use the target.
PRETRAIN_MODES = ("pretrain", "pretrain_supervised", "pretrain_supervise_encoder", "pretrain_simclr")
# These write the encoder file that linear_probe and finetune load. The later run overwrites it.
# SimCLR writes the encoder only. The projection head stays out of that file.
SAVE_ENCODER_MODES = ("pretrain", "pretrain_supervise_encoder", "pretrain_simclr")
# Default when train.py is started with no --mode.
# Choose from the command line instead of editing this list, for example:
#   python train.py --mode pretrain
#   python train.py --mode linear_probe finetune
# pretrain_simclr is not in this list. It writes the same encoder file as pretrain:
#   python train.py --mode pretrain_simclr
#   python train.py --mode linear_probe
MODES = ["pretrain", "linear_probe", "finetune", "end_to_end"]

# SimCLR (model.SimCLR, losses in losses.py). The encoder is still LATENT_DIM wide.
# The projector is thrown away after pretraining. The other batch rows are negatives,
# so this batch is larger than BATCH_SIZE.
# CONTRASTIVE_LOSS: "nt_xent" or "triplet".
TEMPERATURE = 0.07  # NT-Xent
TRIPLET_MARGIN = 0.2  # triplet loss
PROJECTOR_DIM = 128  # g(h) width; its hidden layer is 4 times this, as in the tutorial
SIMCLR_BATCH_SIZE = 256
CONTRASTIVE_LOSS = "nt_xent"

SOURCE_DATASET = "MNIST"
TARGET_DATASET = "FashionMNIST"
N_TARGET_TRAIN = None  # None = all 50,000 Fashion-MNIST training images
TARGET_NUM_EPOCHS = 3
FINETUNE_LEARNING_RATE = 0.0001  # smaller lr, so we don't destroy the pretrained features
ENCODER_PATH = "mnist_encoder.pt"  # written by pretrain, pretrain_supervise_encoder, and pretrain_simclr
