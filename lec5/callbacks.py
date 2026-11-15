from pytorch_lightning.callbacks import EarlyStopping, Callback, TQDMProgressBar


class ColabProgressBar(TQDMProgressBar):
    """tqdm bar without the separate validation bar.

    The validation bar is drawn on a second line, and Colab can't redraw a second
    line in place, so every update would be printed as a new line.
    Validation metrics still appear at the end of the main epoch bar.
    """

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


class EpochMetricsCallback(Callback):
    """Logs one point per epoch under "epoch/...", with the epoch number as the x-axis.

    Lightning's own charts use the training step as the x-axis. Pretraining uses the
    full training set; the target runs use fewer labeled images, so their curves
    don't line up. With the epoch as the x-axis, all modes can be compared on one chart.
    """

    def _log_metrics(self, trainer, names):
        metrics = {}
        for name in names:
            # with on_step=True Lightning stores the epoch value as "<name>_epoch"
            value = trainer.callback_metrics.get(f"{name}_epoch", trainer.callback_metrics.get(name))
            if value is not None:
                metrics[f"epoch/{name}"] = value.item()

        if metrics and trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_metrics(
            trainer,
            [
                "train_loss",
                "train_accuracy",
                "train_f1_score",
                "train_pos_sim",
                "train_neg_sim",
                "train_acc_top1",
                "train_acc_top5",
                "train_acc_mean_pos",
                "train_active",
            ],
        )

    def on_validation_end(self, trainer, pl_module):
        # Skip the sanity check and the extra trainer.validate() after training
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        self._log_metrics(
            trainer,
            [
                "val_loss",
                "val_accuracy",
                "val_pos_sim",
                "val_neg_sim",
                "val_acc_top1",
                "val_acc_top5",
                "val_acc_mean_pos",
                "val_active",
            ],
        )
