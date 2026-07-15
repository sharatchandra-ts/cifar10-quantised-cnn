"""Export one deterministic CIFAR-10 image as unsigned 8-bit hardware inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import ImageOps
from torchvision.datasets import CIFAR10

from utils.config import load_config


def pack_words(values: np.ndarray, word_width_bits: int) -> list[str]:
    """Pack byte stream into little-endian hexadecimal memory words."""
    if word_width_bits % 8:
        raise ValueError("word_width_bits must be a multiple of 8")

    bytes_per_word = word_width_bits // 8
    words = []
    for start in range(0, len(values), bytes_per_word):
        chunk = values[start : start + bytes_per_word]
        padded = np.pad(chunk, (0, bytes_per_word - len(chunk)))
        words.append("".join(f"{value:02x}" for value in padded[::-1]))
    return words


def write_hex(values: np.ndarray, path: Path) -> None:
    path.write_text("".join(f"{value:02x}\n" for value in values), encoding="utf-8")


def write_coe(words: list[str], path: Path) -> None:
    lines = ["memory_initialization_radix = 16;", "memory_initialization_vector ="]
    lines.extend(f"{word}{',' if index < len(words) - 1 else ';'}" for index, word in enumerate(words))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_assembly(words: list[str], symbol: str, path: Path) -> None:
    lines = [
        f".global {symbol}",
        ".section .rodata",
        ".align 2",
        "",
        f"{symbol}:",
        *(f"    .word 0x{word}" for word in words),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = CIFAR10(root=config.data.root, train=False, download=True)
    image, label = dataset[config.export.sample_index]
    if config.model.greyscale:
        image = ImageOps.grayscale(image)

    values = np.asarray(image, dtype=np.uint8).reshape(-1)
    output_dir = Path(config.export.image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = config.export.image_basename
    words = pack_words(values, config.export.word_width_bits)

    hex_path = output_dir / f"{stem}.hex"
    coe_path = output_dir / f"{stem}.coe"
    assembly_path = output_dir / f"{stem}.S"
    metadata_path = output_dir / f"{stem}.json"
    write_hex(values, hex_path)
    write_coe(words, coe_path)
    write_assembly(words, config.export.assembly_symbol, assembly_path)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": "CIFAR-10",
                "split": "test",
                "index": config.export.sample_index,
                "label": int(label),
                "shape": list(np.asarray(image).shape),
                "dtype": "uint8",
                "byte_order": "one byte per .hex line; little-endian within .coe/.S words",
            }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(values)} unsigned 8-bit pixels from CIFAR-10 test index {config.export.sample_index}")
    print(f"  HEX: {hex_path}")
    print(f"  COE: {coe_path}")
    print(f"  ASM: {assembly_path}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
