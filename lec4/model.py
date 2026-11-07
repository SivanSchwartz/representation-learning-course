import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torchvision

from losses import nt_xent_loss, triplet_loss


class NN(pl.LightningModule):
    """Supervised ResNet-18. Encoder and classification head are trained together.

    This is the previous model: pretrain on MNIST, or train from scratch on
    Fashion-MNIST. Cross-entropy on the labels. Nothing is loaded or frozen.
    """

    def __init__(self, learning_rate, num_classes):
        super().__init__()
        self.save_hyperparameters()  # shows lr in TensorBoard's HPARAMS tab
        self.lr = learning_rate

        # Encoder f_theta: a standard ResNet-18, random init (no downloaded weights).
        self.encoder = torchvision.models.resnet18(weights=None)
        # The stock layer expects 3 RGB channels.
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # [B, 1, 28, 28] -> [B, 64, 14, 14]
        num_features = self.encoder.fc.in_features  # 512
        # Remaining ResNet-18 blocks: [B, 64, 14, 14] -> [B, 512].
        # Identity drops ResNet's own 1000-way layer, so the encoder output is z.
        self.encoder.fc = nn.Identity()

        self.head = nn.Linear(num_features, num_classes)  # h_phi: [B, 512] -> [B, num_classes]

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        z = self.encoder(x)  # [B, 1, 28, 28] -> [B, 512]
        return self.head(z)  # [B, 512] -> [B, num_classes]

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
        x, y = batch
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
        self.log_dict(
            {"test_loss": loss, "test_accuracy": accuracy},
            on_step=False, on_epoch=True,
        )
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
        return optim.Adam(self.parameters(), lr=self.lr)


class AutoEncoder(pl.LightningModule):
    """Convolutional autoencoder. Encoder and decoder are defined in this class.

    Two stride-2 convolutions halve a 28x28 image twice (28 -> 14 -> 7). A 3x3
    kernel with padding 1 does that exactly, so no separate pooling layer is needed.
    Channels go 1 -> 32 -> 64: each time the map shrinks, it gets wider, which is
    the usual way to keep some capacity. A linear layer then compresses 64*7*7
    numbers into a short vector z. That vector is the bottleneck, and it is also
    what a later linear classifier can read.

    The decoder is the mirror image. A linear layer expands z back to 64x7x7, and
    two transposed convolutions invert the two stride-2 convolutions (7 -> 14 -> 28).
    output_padding=1 is what makes those sizes land on 14 and 28 rather than 13 and 27.
    Sigmoid keeps every pixel in [0, 1], which is the range ToTensor produces.
    The loss is mean squared error between the input and that reconstruction.
    """

    def __init__(self, learning_rate, latent_dim):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 1, 28, 28] -> [B, 32, 14, 14]
            nn.ReLU(),  # [B, 32, 14, 14] -> [B, 32, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 14, 14] -> [B, 64, 7, 7]
            nn.ReLU(),  # [B, 64, 7, 7] -> [B, 64, 7, 7]
            nn.Flatten(),  # [B, 64, 7, 7] -> [B, 3136]
            nn.Linear(64 * 7 * 7, latent_dim),  # [B, 3136] -> [B, latent_dim]
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),  # [B, latent_dim] -> [B, 3136]
            nn.ReLU(),  # [B, 3136] -> [B, 3136]
            nn.Unflatten(1, (64, 7, 7)),  # [B, 3136] -> [B, 64, 7, 7]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 7, 7] -> [B, 32, 14, 14]
            nn.ReLU(),  # [B, 32, 14, 14] -> [B, 32, 14, 14]
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, 14, 14] -> [B, 1, 28, 28]
            nn.Sigmoid(),  # [B, 1, 28, 28] -> [B, 1, 28, 28], values in (0, 1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)     # [B, 1, 28, 28] -> [B, latent_dim]
        return self.decoder(z)  # [B, latent_dim] -> [B, 1, 28, 28]

    def training_step(self, batch, batch_idx):
        loss, recon, x = self._common_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self._log_reconstructions(x, recon, "train_reconstructions")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, x = self._common_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0 and not self.trainer.sanity_checking:
            self._log_reconstructions(x, recon, "val_reconstructions")
        return loss

    def test_step(self, batch, batch_idx):
        loss, recon, x = self._common_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        if batch_idx == 0:
            self._log_reconstructions(x, recon, "test_reconstructions")
        return loss

    def _common_step(self, batch):
        x, _ = batch  # x: [B, 1, 28, 28]; the label is unused
        recon = self.forward(x)
        loss = self.loss_fn(recon, x)
        return loss, recon, x

    def _log_reconstructions(self, x, recon, tag):
        """Top row: inputs. Bottom row: reconstructions."""
        if self.logger is None:
            return
        n = min(8, x.size(0))
        images = torch.cat([x[:n], recon[:n]]).detach().float().cpu()
        grid = torchvision.utils.make_grid(images, nrow=n)
        self.logger.experiment.add_image(tag, grid, self.global_step)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class EncoderClassifier(pl.LightningModule):
    """Pretrained encoder plus a new linear head. Used for linear probe and finetune.

    The encoder layers match AutoEncoder.encoder. encoder_path is loaded here,
    so the head is the only part that starts from random weights.
    freeze_encoder=True trains only the head. False trains the encoder and the head.
    """

    def __init__(self, learning_rate, num_classes, latent_dim, encoder_path, freeze_encoder=False):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.freeze_encoder = freeze_encoder

        # Same layers, same order as AutoEncoder.encoder.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 1, 28, 28] -> [B, 32, 14, 14]
            nn.ReLU(),  # [B, 32, 14, 14] -> [B, 32, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 14, 14] -> [B, 64, 7, 7]
            nn.ReLU(),  # [B, 64, 7, 7] -> [B, 64, 7, 7]
            nn.Flatten(),  # [B, 64, 7, 7] -> [B, 3136]
            nn.Linear(64 * 7 * 7, latent_dim),  # [B, 3136] -> [B, latent_dim]
        )
        # Weights from AutoEncoder pretraining. The head below is not in this file.
        state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state_dict)

        self.head = nn.Linear(latent_dim, num_classes)  # [B, latent_dim] -> [B, num_classes]

        # eval() before freezing: a frozen encoder must not update BatchNorm stats
        # or dropout. train() below keeps that, because Lightning calls train()
        # at the start of every epoch and would otherwise undo it.
        if self.freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x):
        z = self.encoder(x)  # [B, 1, 28, 28] -> [B, latent_dim]
        return self.head(z)  # [B, latent_dim] -> [B, num_classes]

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
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        x, y = batch
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
        self.log_dict(
            {"test_loss": loss, "test_accuracy": accuracy},
            on_step=False, on_epoch=True,
        )
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch  # x: [B, 1, 28, 28]
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        return loss, outputs, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        # Linear probing: only the head is trainable, so only the head is optimized.
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(params, lr=self.lr)


class SupervisedEncoder(pl.LightningModule):
    """Convolutional encoder plus a linear head, both trained from scratch.

    Same encoder layers as AutoEncoder, with cross-entropy instead of a decoder.
    This sits beside NN: NN is ResNet-18, this is the small encoder.
    """

    def __init__(self, learning_rate, num_classes, latent_dim):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate

        # Same layers, same order as AutoEncoder.encoder.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 1, 28, 28] -> [B, 32, 14, 14]
            nn.ReLU(),  # [B, 32, 14, 14] -> [B, 32, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 14, 14] -> [B, 64, 7, 7]
            nn.ReLU(),  # [B, 64, 7, 7] -> [B, 64, 7, 7]
            nn.Flatten(),  # [B, 64, 7, 7] -> [B, 3136]
            nn.Linear(64 * 7 * 7, latent_dim),  # [B, 3136] -> [B, latent_dim]
        )
        self.head = nn.Linear(latent_dim, num_classes)  # [B, latent_dim] -> [B, num_classes]

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        z = self.encoder(x)  # [B, 1, 28, 28] -> [B, latent_dim]
        return self.head(z)  # [B, latent_dim] -> [B, num_classes]

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
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        x, y = batch
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
        self.log_dict(
            {"test_loss": loss, "test_accuracy": accuracy},
            on_step=False, on_epoch=True,
        )
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch  # x: [B, 1, 28, 28]
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        return loss, outputs, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class SimCLR(pl.LightningModule):
    """Contrastive pretraining on two views of each image.

    The encoder is the same small conv encoder as AutoEncoder, so the file
    written by train.py loads in EncoderClassifier. Each view is encoded to h
    and projected to z. The loss is nt_xent_loss or triplet_loss. The projector is not saved.
    """

    def __init__(self, learning_rate, latent_dim, projector_dim, temperature, margin, loss_name):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.temperature = temperature
        self.margin = margin
        self.loss_name = loss_name

        # Same layers, same order as AutoEncoder.encoder.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 1, 28, 28] -> [B, 32, 14, 14]
            nn.ReLU(),  # [B, 32, 14, 14] -> [B, 32, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 14, 14] -> [B, 64, 7, 7]
            nn.ReLU(),  # [B, 64, 7, 7] -> [B, 64, 7, 7]
            nn.Flatten(),  # [B, 64, 7, 7] -> [B, 3136]
            nn.Linear(64 * 7 * 7, latent_dim),  # [B, 3136] -> [B, latent_dim]
        )
        # g(h): Linear -> ReLU -> Linear, with a 4x wider hidden layer.
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 4 * projector_dim),  # [B, latent_dim] -> [B, 4 * projector_dim]
            nn.ReLU(inplace=True),  # [B, 4 * projector_dim] -> [B, 4 * projector_dim]
            nn.Linear(4 * projector_dim, projector_dim),  # [B, 4 * projector_dim] -> [B, projector_dim]
        )

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, "train")
        if batch_idx % 100 == 0:
            (xi, xj), _ = batch
            self._log_views(xi, xj)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def _common_step(self, batch, mode):
        (xi, xj), _ = batch  # two augmented views, each [B, 1, 28, 28]; the label is unused

        # representations
        h1 = self.encoder(xi)  # [B, 1, 28, 28] -> [B, latent_dim]
        h2 = self.encoder(xj)  # [B, 1, 28, 28] -> [B, latent_dim]

        z1 = self.projector(h1)  # [B, latent_dim] -> [B, projector_dim]
        z2 = self.projector(h2)  # [B, latent_dim] -> [B, projector_dim]

        if self.loss_name == "nt_xent":
            loss = nt_xent_loss(z1, z2, self.temperature)
        elif self.loss_name == "triplet":
            loss = triplet_loss(z1, z2, self.margin)
        else:
            raise ValueError(f"Unknown loss {self.loss_name!r}. Use 'nt_xent' or 'triplet'.")

        self.log(
            f"{mode}_loss",
            loss,
            on_step=(mode == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        # How close a positive pair is, compared with the other images in the batch.
        stats = self._pair_stats(z1, z2)
        self.log(
            f"{mode}_pos_sim",
            stats.pop("pos_sim"),
            on_step=(mode == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {f"{mode}_{name}": value for name, value in stats.items()},
            on_step=(mode == "train"),
            on_epoch=True,
        )
        return loss

    def _pair_stats(self, z1, z2):
        """Scalars for TensorBoard. They are not part of the loss."""
        z1 = F.normalize(z1.detach(), dim=-1)
        z2 = F.normalize(z2.detach(), dim=-1)
        # Cosine similarity of every anchor in z1 with every view in z2.
        pair = torch.mm(z1, z2.t())  # [B, B]
        eye = torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)
        n_negatives = pair.shape[0] * (pair.shape[0] - 1)
        stats = {
            "pos_sim": pair.diag().mean(),
            "neg_sim": pair.masked_fill(eye, 0).sum() / n_negatives,
        }

        if self.loss_name == "nt_xent":
            # Rank of the matching view among the other 2B - 1 rows. 0 means it is the nearest.
            out = torch.cat([z1, z2], dim=0)
            sim = torch.mm(out, out.t())
            n = sim.shape[0]
            self_mask = torch.eye(n, dtype=torch.bool, device=sim.device)
            index = torch.arange(n, device=sim.device)
            positive = sim[index, (index + n // 2) % n]
            rank = ((sim > positive.unsqueeze(1)) & ~self_mask).sum(dim=1)
            stats["acc_top1"] = (rank == 0).float().mean()
            stats["acc_top5"] = (rank < 5).float().mean()
            stats["acc_mean_pos"] = (rank + 1).float().mean()
        elif self.loss_name == "triplet":
            # Share of negatives that still violate the margin.
            dist = (2 - 2 * pair).clamp(min=0)
            positive = dist.diag().unsqueeze(1)
            violate = (positive - dist + self.margin > 0) & ~eye
            stats["active"] = violate.float().sum() / (~eye).float().sum()
        return stats

    def _log_views(self, xi, xj):
        """Top row: xi. Bottom row: xj."""
        if self.logger is None:
            return
        n = min(8, xi.size(0))
        images = torch.cat([xi[:n], xj[:n]]).detach().float().cpu()
        grid = torchvision.utils.make_grid(images, nrow=n)
        self.logger.experiment.add_image("contrastive_views", grid, self.global_step)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
