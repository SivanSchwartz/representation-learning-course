import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision

class NN(pl.LightningModule):
    def __init__(self, learning_rate, num_classes, freeze_encoder=False):
        super().__init__()
        self.save_hyperparameters()  # shows lr / freeze_encoder in TensorBoard's HPARAMS tab
        self.lr = learning_rate
        self.freeze_encoder = freeze_encoder

        # Encoder f_theta: a standard ResNet-18, random init (no downloaded weights).
        # Only change: grayscale images -> 1 input channel instead of 3 (RGB).
        self.encoder = torchvision.models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.encoder.fc.in_features  # 512
        self.encoder.fc = nn.Identity()  # remove ResNet's own head -> encoder outputs z (512-dim)

        # Head h_phi: a linear classifier for the current task
        self.head = nn.Linear(num_features, num_classes)

        # Linear probing: the encoder is not traiined, only the head
        if self.freeze_encoder:
            self.encoder.eval()  # keep BatchNorm/dropout layers in eval mode
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        z = self.encoder(x)  # representation  z = f_theta(x)
        x = self.head(z)     # class scores    h_phi(z)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=True,   # curve that updates during training (x-axis = step)
            on_epoch=True,  # one value per epoch
            prog_bar=True,
        )
        x,y = batch
        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("train_images", grid, self.global_step)

        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy},
            on_step=False, on_epoch=True, prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        self.log_dict({"test_loss": loss, "test_accuracy": accuracy}, on_step=False, on_epoch=True)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch  # x: [B, 1, 28, 28] - a CNN keeps the image shape (no flattening)
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        return loss, outputs, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        # Optimize only trainable parameters (linear probing -> only the head)
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(params, lr=self.lr)
