"""Augmentations used for MNIST training."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.datasets import MNIST

def build_mnist_transforms():
    train_transform = v2.Compose(
        [
            transforms.ToImageTensor(),
            v2.RandomApply(
                [
                    v2.RandomInvert(p=1.0),
                    v2.RandomGaussianNoise(mean=0.0, std=0.1, p=1.0),
                    v2.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
                ],
                p=0.5,
            ),

        ]
    )
    eval_transform = v2.Compose([v2.ToImageTensor()])
    return train_transform, eval_transform
