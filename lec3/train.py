import time
import torch
import pytorch_lightning as pl
from datamodule import MnistDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping, ColabProgressBar, EpochMetricsCallback
from pytorch_lightning.loggers import TensorBoardLogger
from utils import parse_args, build_model

torch.set_float32_matmul_precision("medium")  # to make lightning happy
RUN_ID = time.strftime("%Y-%m-%d_%H-%M-%S")

# pretrain and pretrain_supervised train on the source dataset. The others use the target.
PRETRAIN_MODES = ("pretrain", "pretrain_supervised")


def run(mode):
    """Train and evaluate one experiment. mode: see config.MODES."""
    pl.seed_everything(42)  # same initialization and data order -> fair comparison

    logger = TensorBoardLogger("tb_logs", name=RUN_ID, version=mode, default_hp_metric=False)
    model = build_model(mode)

    if mode in PRETRAIN_MODES:
        dataset_name, n_train, num_epochs = config.SOURCE_DATASET, None, config.NUM_EPOCHS
    else:
        dataset_name, n_train, num_epochs = config.TARGET_DATASET, config.N_TARGET_TRAIN, config.TARGET_NUM_EPOCHS

    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        dataset_name=dataset_name,
        n_train=n_train,
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
    print(
        f"Training mode is: {mode}, on {dataset_name} for {num_epochs} epochs, "
        f"batch size {config.BATCH_SIZE}, n_train={n_train}"
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    results = trainer.test(model, dm)

    if mode == "pretrain":
        # Keep the encoder. The decoder is not saved; EncoderClassifier has its own head.
        torch.save(model.encoder.state_dict(), config.ENCODER_PATH)
        return results[0]["test_loss"]
    return results[0]["test_accuracy"]


if __name__ == "__main__":
    args = parse_args()
    test_results = {}
    for mode in args.mode:
        test_results[mode] = run(mode)

    print("\nTest results:")
    for mode, value in test_results.items():
        if mode == "pretrain":
            print(f"  {mode:<20} ({config.SOURCE_DATASET}): reconstruction MSE {value:.4f}")
        else:
            dataset_name = config.SOURCE_DATASET if mode in PRETRAIN_MODES else config.TARGET_DATASET
            print(f"  {mode:<20} ({dataset_name}): accuracy {value:.3f}")
