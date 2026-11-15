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

# One model name. The dataset is not chosen here: pass --dataset mnist or --dataset fashionmnist.
#   python train.py
#   python train.py --mode linear_probe --dataset fashionmnist
# --mode overrides MODEL_NAME with one of these (see model.py):
#   "pretrain"                    : AutoEncoder - reconstruction, saves the encoder
#   "pretrain_supervised"         : NN - ResNet-18 trained with cross-entropy
#   "pretrain_supervise_encoder"  : SupervisedEncoder - small encoder + head from scratch, saves the encoder
#   "pretrain_simclr"             : SimCLR - contrastive loss on two views, saves the encoder (not the projector)
#   "pretrain_moco"               : MoCo - momentum encoder and a queue of negatives, saves the query encoder
#   "linear_probe"                : EncoderClassifier - that encoder FROZEN + a new head
#   "finetune"                    : EncoderClassifier - that encoder + a new head, all weights
#   "end_to_end"                  : NN - ResNet-18 from scratch
# Pretrain modes train the representation. The others train a classifier on the same dataset.
PRETRAIN_MODES = ("pretrain", "pretrain_supervised", "pretrain_supervise_encoder", "pretrain_simclr", "pretrain_moco")
# These write the encoder file that linear_probe and finetune load. A later run overwrites it.
# SimCLR and MoCo write the encoder only. The projection head stays out of that file.
SAVE_ENCODER_MODES = ("pretrain", "pretrain_supervise_encoder", "pretrain_simclr", "pretrain_moco")
MODEL_NAME = "pretrain_moco"

# SimCLR (model.SimCLR, losses in losses.py). The encoder is still LATENT_DIM wide.
# The projector is thrown away after pretraining. The other batch rows are negatives,
# so this batch is larger than BATCH_SIZE.
# CONTRASTIVE_LOSS: "nt_xent" or "triplet".
TEMPERATURE = 0.07  # NT-Xent, and the MoCo softmax temperature
TRIPLET_MARGIN = 0.2  # triplet loss
PROJECTOR_DIM = 128  # g(h) width; its hidden layer is 4 times this, as in the tutorial
SIMCLR_BATCH_SIZE = 256
CONTRASTIVE_LOSS = "nt_xent"

# MoCo (model.MoCo). Negatives come from the queue, so the batch stays at BATCH_SIZE.
# MOCO_K must be a multiple of BATCH_SIZE. MOCO_M is how slowly the key encoder follows the query encoder.
MOCO_K = 4096
MOCO_M = 0.999

# Set by --dataset in utils.parse_args. Default is MNIST.
DATASET = "MNIST"
N_TARGET_TRAIN = None  # None = all training images, for linear_probe, finetune, and end_to_end
TARGET_NUM_EPOCHS = 3
FINETUNE_LEARNING_RATE = 0.0001  # smaller lr, so we don't destroy the pretrained features
ENCODER_PATH = "mnist_encoder.pt"  # parse_args rewrites this from --dataset
