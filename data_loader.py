import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    greyscale: bool = True,
    num_workers: int = 2,
    val_split: float = 0.1,
):
    # ── Test transform — NO augmentation ─────────────────────────────────
    # Augmentation is only for training. Applying it to test/val gives
    # unreliable accuracy measurements (the model sees different images
    # each evaluation run).
    if greyscale:
        test_transform = v2.Compose([
            v2.ToImage(),
            v2.Grayscale(num_output_channels=1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4734,), (0.2516,)),   # CIFAR-10 greyscale stats
        ])
        # ── Train transform — with augmentation ──────────────────────────
        # RandomCrop(32, padding=4): pads 4px on each side then crops back
        #   to 32×32 — simulates slightly shifted viewpoints, +2-3% acc
        # RandomHorizontalFlip: randomly mirrors image — +1-2% acc
        # These are the two augmentations that consistently help at this scale.
        # ColorJitter is excluded because we're greyscale.
        train_transform = v2.Compose([
            v2.ToImage(),
            v2.Grayscale(num_output_channels=1),
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4734,), (0.2516,)),
        ])
        print("Training on greyscale dataset")
    else:
        test_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),  # CIFAR-10 RGB stats
        ])
        train_transform = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
        ])
        print("Training on RGB dataset")

    # Train and val share trainset but val uses test_transform (no augmentation)
    trainset_aug  = torchvision.datasets.CIFAR10(root=root, train=True,
                                                  download=True, transform=train_transform)
    trainset_plain = torchvision.datasets.CIFAR10(root=root, train=True,
                                                   download=True, transform=test_transform)
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=test_transform)

    # Same deterministic split for both augmented and plain trainsets
    n_total   = len(trainset_aug)
    train_size = int((1 - val_split) * n_total)
    val_size   = n_total - train_size
    generator  = torch.Generator().manual_seed(42)

    trainset_split, _ = random_split(trainset_aug,   [train_size, val_size], generator=generator)
    _, valset_split   = random_split(trainset_plain, [train_size, val_size], generator=generator)

    trainloader = DataLoader(trainset_split, batch_size=batch_size,
                             shuffle=True,  num_workers=num_workers, pin_memory=True)
    valloader   = DataLoader(valset_split,   batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    testloader  = DataLoader(testset,        batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, valloader, testloader