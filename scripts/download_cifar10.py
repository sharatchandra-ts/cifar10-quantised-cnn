"""Download CIFAR-10 without starting a training run."""

from __future__ import annotations

import argparse

from torchvision.datasets import CIFAR10

from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    args = parser.parse_args()

    config = load_config(args.config)
    root = config.data.root
    CIFAR10(root=root, train=True, download=True)
    CIFAR10(root=root, train=False, download=True)
    print(f"CIFAR-10 train and test sets are ready in {root}/")


if __name__ == "__main__":
    main()
