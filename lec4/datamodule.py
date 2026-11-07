import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Subset
import pytorch_lightning as pl

from augmentations import ContrastiveTransformations, build_mnist_transforms, build_simclr_transform


class MnistDataModule(pl.LightningDataModule):
    """Works for MNIST and Fashion-MNIST (same 28x28 grayscale format)."""

    def __init__(self, data_dir, batch_size, num_workers, dataset_name="MNIST", n_train=None, two_views=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_cls = getattr(datasets, dataset_name)  # datasets.MNIST or datasets.FashionMNIST
        self.n_train = n_train  # None = use all training images
        self.two_views = two_views  # SimCLR: each item is (xi, xj) instead of one image

    def prepare_data(self):
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage):
        if self.two_views:
            # xi and xj are two draws of the SimCLR augmentation. Val and test
            # use that same policy, so their loss is the contrastive loss too.
            train_transform = ContrastiveTransformations(build_simclr_transform())
            eval_transform = train_transform
        else:
            train_transform, eval_transform = build_mnist_transforms()
        # Two copies of the training set. random_split subsets share one dataset,
        # so they cannot hold different transforms. One-view runs keep augmentation
        # off the val set. Contrastive pretraining points both copies at (xi, xj).
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
            # The contrastive loss treats the rest of the batch as negatives.
            # A short last batch would be a different number of negatives.
            drop_last=self.two_views,
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
