import torch
from torchvision.transforms import v2


class ContrastiveTransformations:
    """Apply one transform twice and return the two views.

    Each call samples new random parameters, so xi and xj differ.
    The base transform is whatever policy you pass in.
    """

    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        xi = self.base_transforms(x)
        xj = self.base_transforms(x)
        return xi, xj


def build_simclr_transform():
    """One SimCLR view of a 28x28 grayscale image.

    Same steps as the notebook: crop, horizontal flip, color jitter, grayscale,
    and a gaussian blur. Blur kernels have to be odd, and 10% of 28 is 2, so
    the kernel is 3. Normalization matches the other MNIST transforms.
    """
    color_jitter = v2.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(size=28),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([color_jitter], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            v2.Normalize((0.1307,), (0.3081,)),
        ]
    )


def build_mnist_transforms():
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.Normalize((0.1307,), (0.3081,)),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return train_transform, eval_transform
