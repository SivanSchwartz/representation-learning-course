from pytorch_lightning.callbacks import EarlyStopping, Callback


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


class EpochMetricsCallback(Callback):
    def _log_metrics(self, trainer, names):
        metrics = {
            f"epoch/{name}": trainer.callback_metrics[name]
            for name in names
            if name in trainer.callback_metrics
        }

        if metrics and trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_metrics(
            trainer,
            ["train_loss", "train_accuracy", "train_f1_score"],
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, ["val_loss"])

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, ["test_loss"])
