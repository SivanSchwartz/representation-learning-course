import torch
from torchvision.transforms import v2


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
