# Training hyperparameters
NUM_CLASSES = 10  # MNIST and Fashion-MNIST both have 10 classes (but different meanings!)
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "auto"  # GPU on Colab, CPU on this Mac
DEVICES = 1
PRECISION = "16-mixed"  # same as 16 (the old name); falls back to bf16 on CPU

# Transfer learning
# Experiments to run, in this order (see train.py):
#   "pretrain"     : step 1 - ResNet-18 from scratch on MNIST (source task), saves the encoder
#   "linear_probe" : step 2 - MNIST encoder FROZEN + new linear head, trained on Fashion-MNIST
#   "finetune"     : step 2 - MNIST encoder + new linear head, ALL weights trained on Fashion-MNIST
#   "end_to_end"   : baseline - ResNet-18 from scratch on Fashion-MNIST (no transfer)
# Run one experiment, e.g. ["linear_probe"] (needs a saved encoder),
# or chain them to see step 1 and step 2 one after the other.
MODES = ["pretrain", "linear_probe", "finetune", "end_to_end"]

SOURCE_DATASET = "MNIST"
TARGET_DATASET = "FashionMNIST"
N_TARGET_TRAIN = 1000  # labeled target images to train on (None = all 50,000)
TARGET_NUM_EPOCHS = 3  # small training set -> more epochs
FINETUNE_LEARNING_RATE = 0.0001  # smaller lr, so we don't destroy the pretrained features
ENCODER_PATH = "mnist_resnet18_encoder.pt"  # where step 1 saves the encoder
