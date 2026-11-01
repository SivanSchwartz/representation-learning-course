import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Subset
import pytorch_lightning as pl

from augmentations import build_mnist_transforms


class MnistDataModule(pl.LightningDataModule):
    """Works for MNIST and Fashion-MNIST (same 28x28 grayscale format)."""

    def __init__(self, data_dir, batch_size, num_workers, dataset_name="MNIST", n_train=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_cls = getattr(datasets, dataset_name)  # datasets.MNIST or datasets.FashionMNIST
        self.n_train = n_train  # None = use all training images

    def prepare_data(self):
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage):
        train_transform, eval_transform = build_mnist_transforms()
        # Two copies of the training set so train can be augmented and val cannot.
        # random_split subsets share one dataset, so they cannot hold different transforms.
        train_full = self.dataset_cls(
            root=self.data_dir,
            train=True,
            transform=train_transform,
            download=False,
        )
        val_full = self.dataset_cls(
            root=self.data_dir,
            train=True,
            transform=eval_transform,
            download=False,
        )
        # Fixed seed -> every experiment (and every call to setup) gets the same split
        generator = torch.Generator().manual_seed(42)
        self.train_ds, val_split = random_split(train_full, [50000, 10000], generator=generator)
        self.val_ds = Subset(val_full, val_split.indices)
        # Keep only n_train labeled images (simulates a small labeled dataset)
        if self.n_train is not None:
            self.train_ds = Subset(self.train_ds, range(self.n_train))

        self.test_ds = self.dataset_cls(
            root=self.data_dir,
            train=False,
            transform=eval_transform,
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
