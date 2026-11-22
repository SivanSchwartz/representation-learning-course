import time
import torch
import pytorch_lightning as pl
from datamodule import MnistDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping, ColabProgressBar, EpochMetricsCallback
from pytorch_lightning.loggers import TensorBoardLogger
from utils import parse_args, build_model
from model import NN
torch.set_float32_matmul_precision("medium")  # to make lightning happy
RUN_ID = time.strftime("%Y-%m-%d_%H-%M-%S")


def run(mode):
    """Train and evaluate one experiment. mode is the required --mode argument."""
    pl.seed_everything(42)  # same initialization and data order -> fair comparison

    logger = TensorBoardLogger("tb_logs", name=RUN_ID, version=mode, default_hp_metric=False)
    model = build_model(mode)

    # Pretraining uses every training image. The classifier modes can use a subset.
    # The dataset itself is --dataset, for every mode.
    if mode in config.PRETRAIN_MODES:
        n_train, num_epochs = None, config.NUM_EPOCHS
    else:
        n_train, num_epochs = config.N_TARGET_TRAIN, config.TARGET_NUM_EPOCHS
    dataset_name = config.DATASET

    # SimCLR compares each image with the rest of its batch, so it uses a larger batch.
    # MoCo reads negatives from a queue, so it keeps BATCH_SIZE.
    # BYOL has no negatives, so it also keeps BATCH_SIZE.
    # All three ask the dataset for two augmented views. Every other mode is unchanged.
    if mode == "pretrain_simclr":
        batch_size, two_views = config.SIMCLR_BATCH_SIZE, True
    elif mode in ("pretrain_moco", "pretrain_byol"):
        batch_size, two_views = config.BATCH_SIZE, True
    else:
        batch_size, two_views = config.BATCH_SIZE, False

    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        dataset_name=dataset_name,
        n_train=n_train,
        two_views=two_views,
    )

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=num_epochs,
        precision=config.PRECISION,
        log_every_n_steps=10,  # default is 50: short runs would get no step points
        callbacks=[
            MyPrintingCallback(),
            EarlyStopping(monitor="val_loss"),
            ColabProgressBar(refresh_rate=20),
            EpochMetricsCallback(),
        ],
    )
    print(model)
    details = (
        f"Training mode is: {mode}, on {dataset_name} for {num_epochs} epochs, "
        f"batch size {batch_size}, n_train={n_train}"
    )
    if mode == "pretrain_simclr":
        details += f", loss {config.CONTRASTIVE_LOSS}"
    if mode == "pretrain_moco":
        details += f", queue {config.MOCO_K}, momentum {config.MOCO_M}"
    if mode == "pretrain_byol":
        details += f", momentum {config.BYOL_M}"
    print(details)
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    results = trainer.test(model, dm)

    if mode in config.SAVE_ENCODER_MODES:
        # Keep the encoder. The decoder, the projector, or the classification head is not saved.
        torch.save(model.encoder.state_dict(), config.ENCODER_PATH)
    if mode in ("pretrain", "pretrain_simclr", "pretrain_moco", "pretrain_byol"):
        return results[0]["test_loss"]
    return results[0]["test_accuracy"]


if __name__ == "__main__":
    args = parse_args()
    value = run(args.mode)

    print("\nTest results:")
    if args.mode == "pretrain":
        print(f"  {args.mode:<28} ({config.DATASET}): reconstruction MSE {value:.4f}")
    elif args.mode == "pretrain_simclr":
        print(f"  {args.mode:<28} ({config.DATASET}): {config.CONTRASTIVE_LOSS} {value:.4f}")
    elif args.mode == "pretrain_moco":
        print(f"  {args.mode:<28} ({config.DATASET}): queue loss {value:.4f}")
    elif args.mode == "pretrain_byol":
        print(f"  {args.mode:<28} ({config.DATASET}): regression loss {value:.4f}")
    else:
        print(f"  {args.mode:<28} ({config.DATASET}): accuracy {value:.3f}")
