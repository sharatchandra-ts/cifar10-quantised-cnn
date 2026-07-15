"""Export layer-1 hardware verification vectors for the configured CIFAR-10 image."""

from __future__ import annotations

import argparse
import copy
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from model import GoldenCNNModel
from quant import QuantBits, fold_bn_pair, prepare_qat, set_fake_quant
from utils.config import load_config


HEX_BYTE = re.compile(r"^[0-9a-fA-F]{2}$")
LAYER_HEADER = re.compile(r"^// (?P<name>\S+)  shape=.* bits=\d+$")


def pack_bytes(values: np.ndarray, word_width_bits: int) -> list[str]:
    if word_width_bits % 8:
        raise ValueError("word_width_bits must be a multiple of 8")

    bytes_per_word = word_width_bits // 8
    words = []
    for start in range(0, len(values), bytes_per_word):
        chunk = values[start : start + bytes_per_word]
        padded = np.pad(chunk, (0, bytes_per_word - len(chunk)))
        words.append("".join(f"{value:02x}" for value in padded[::-1]))
    return words


def write_vector(
    values: np.ndarray,
    stem: str,
    symbol: str,
    directories: dict[str, Path],
    word_width_bits: int,
) -> None:
    byte_values = np.asarray(values, dtype=np.uint8).reshape(-1)
    (directories["hex"] / f"{stem}.hex").write_text(
        "".join(f"{value:02x}\n" for value in byte_values), encoding="utf-8"
    )
    words = pack_bytes(byte_values, word_width_bits)
    coe_lines = ["memory_initialization_radix = 16;", "memory_initialization_vector ="]
    coe_lines.extend(
        f"{word}{',' if index < len(words) - 1 else ';'}"
        for index, word in enumerate(words)
    )
    (directories["coe"] / f"{stem}.coe").write_text(
        "\n".join(coe_lines) + "\n", encoding="utf-8"
    )
    assembly_lines = [
        f".global {symbol}",
        ".section .rodata",
        ".align 2",
        "",
        f"{symbol}:",
        *(f"    .word 0x{word}" for word in words),
        "",
    ]
    (directories["asm"] / f"{stem}.S").write_text(
        "\n".join(assembly_lines), encoding="utf-8"
    )


def write_int16_vector(
    values: np.ndarray,
    stem: str,
    symbol: str,
    directories: dict[str, Path],
) -> None:
    signed_values = np.asarray(values, dtype=np.int16).reshape(-1)
    unsigned_values = signed_values.view(np.uint16)
    hex_values = [f"{value:04x}" for value in unsigned_values]
    (directories["hex"] / f"{stem}.hex").write_text(
        "\n".join(hex_values) + "\n", encoding="utf-8"
    )
    words = []
    for start in range(0, len(unsigned_values), 2):
        low = int(unsigned_values[start])
        high = int(unsigned_values[start + 1]) if start + 1 < len(unsigned_values) else 0
        words.append(f"{(high << 16) | low:08x}")
    coe_lines = ["memory_initialization_radix = 16;", "memory_initialization_vector ="]
    coe_lines.extend(
        f"{word}{',' if index < len(words) - 1 else ';'}"
        for index, word in enumerate(words)
    )
    (directories["coe"] / f"{stem}.coe").write_text(
        "\n".join(coe_lines) + "\n", encoding="utf-8"
    )
    assembly_lines = [
        f".global {symbol}",
        ".section .rodata",
        ".align 2",
        "",
        f"{symbol}:",
        *(f"    .word 0x{word}" for word in words),
        "",
    ]
    (directories["asm"] / f"{stem}.S").write_text(
        "\n".join(assembly_lines), encoding="utf-8"
    )


def layer_one_weights(path: Path) -> np.ndarray:
    values: list[int] = []
    found_layer_one = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        header = LAYER_HEADER.fullmatch(raw_line)
        if header:
            if found_layer_one:
                break
            found_layer_one = header.group("name") == "conv_layers.0"
        elif found_layer_one and HEX_BYTE.fullmatch(raw_line):
            values.append(int(raw_line, 16))

    if not values:
        raise ValueError(f"No layer-1 weights found in {path}")
    return np.asarray(values, dtype=np.uint8)


def layer_one_bias(path: Path) -> np.ndarray:
    values: list[int] = []
    found_layer_one = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("// conv_layers."):
            if found_layer_one:
                break
            found_layer_one = raw_line.startswith("// conv_layers.0 ")
        elif found_layer_one and raw_line.startswith("b  "):
            value = int(raw_line.split()[1], 16)
            values.append(value if value < (1 << 31) else value - (1 << 32))

    if len(values) != 16:
        raise ValueError(f"Expected 16 layer-1 bias values in {path}, found {len(values)}")
    return np.asarray(values, dtype=np.int32)


def test_transform(greyscale: bool):
    if greyscale:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Grayscale(num_output_channels=1),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.4734,), (0.2516,)),
            ]
        )
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def fold_batch_norm(model: GoldenCNNModel) -> GoldenCNNModel:
    export_model = copy.deepcopy(model)
    if not export_model.use_bn:
        return export_model

    for index in range(export_model.num_active_layers):
        quantized_conv = export_model.conv_layers[index]
        quantized_conv.module = fold_bn_pair(quantized_conv.module, export_model.bn_layers[index])
        export_model.bn_layers[index] = nn.Identity()
    return export_model


def as_int8_bytes(values: torch.Tensor, scale: float) -> np.ndarray:
    return (
        (values.detach().cpu() / scale)
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
        .numpy()
        .view(np.uint8)
    )


def layer_one_accumulator(
    input_values: np.ndarray,
    weight_values: np.ndarray,
    bias_values: np.ndarray,
) -> np.ndarray:
    input_tensor = torch.from_numpy(input_values.view(np.int8).astype(np.float32)).reshape(1, 1, 32, 32)
    weight_tensor = torch.from_numpy(weight_values.view(np.int8).astype(np.float32)).reshape(16, 1, 3, 3)
    bias_tensor = torch.from_numpy(bias_values.astype(np.float32))
    accumulator = functional.conv2d(input_tensor, weight_tensor, bias_tensor, padding=1)
    values = accumulator.round().to(torch.int32).numpy().squeeze(0)
    if values.min() < np.iinfo(np.int16).min or values.max() > np.iinfo(np.int16).max:
        raise OverflowError("Layer-1 accumulator does not fit signed INT16")
    return values.astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    parser.add_argument("--quant-bits", default="INT4", choices=["INT4", "INT8"])
    args = parser.parse_args()

    config = load_config(args.config)
    quant_bits = QuantBits[args.quant_bits]
    checkpoint_path = Path("models") / quant_bits.name / f"golden_model_qat_{quant_bits.name}.pt"
    weight_path = Path("models") / quant_bits.name / f"weights_{quant_bits.name}.hex"
    requant_path = Path("models") / quant_bits.name / f"requant_{quant_bits.name}.hex"
    if not checkpoint_path.exists() or not weight_path.exists() or not requant_path.exists():
        raise FileNotFoundError("Run `make train` or `make export` to create the INT4 checkpoint and weights.")

    model = GoldenCNNModel(config)
    prepare_qat(model, quant_bits)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model = fold_batch_norm(model)
    model.eval()
    set_fake_quant(model, enabled=True)

    dataset = CIFAR10(root=config.data.root, train=False, download=True)
    image, _ = dataset[config.export.sample_index]
    input_tensor = test_transform(config.model.greyscale)(image).unsqueeze(0)

    with torch.no_grad():
        input_quantized = model.input_quant(input_tensor)
        layer_one = model.conv_layers[0](input_quantized)
        layer_one = model.bn_layers[0](layer_one)
        output_no_maxpool = model.act(layer_one)
        output_maxpool = model.pool(output_no_maxpool)

    activation_scale = model.act_quant[0].get_scale()
    input_values = as_int8_bytes(input_quantized.squeeze(0), model.input_quant.get_scale())
    weight_values = layer_one_weights(weight_path)
    intermediate_values = layer_one_accumulator(
        input_values,
        weight_values,
        layer_one_bias(requant_path),
    )
    intermediate_values = np.maximum(intermediate_values, 0).astype(np.int16)
    output_root = Path(config.export.hardware_dir) / quant_bits.name
    shutil.rmtree(output_root, ignore_errors=True)
    directories = {name: output_root / name for name in ("asm", "coe", "hex")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    write_vector(
        input_values,
        "input_layer1_qint8",
        "cnn_input_layer1_qint8",
        directories,
        config.export.word_width_bits,
    )
    write_vector(
        weight_values,
        "weights_layer1_int4",
        "cnn_weights_layer1_int4",
        directories,
        config.export.word_width_bits,
    )
    write_int16_vector(
        intermediate_values,
        "output_layer1_intermediate_relu_int16",
        "cnn_output_layer1_intermediate_relu_int16",
        directories,
    )
    write_vector(
        as_int8_bytes(output_maxpool.squeeze(0), activation_scale),
        "output_layer1_full_qint8",
        "cnn_output_layer1_full_qint8",
        directories,
        config.export.word_width_bits,
    )
    print(f"Exported layer-1 verification vectors to {output_root}")
    print("  input:  1 x 32 x 32 signed INT8")
    print("  weights: 16 x 1 x 3 x 3 signed INT4 stored in bytes")
    print("  intermediate output: 16 x 32 x 32 signed INT16 accumulator after ReLU")
    print("  full output:         16 x 16 x 16 signed INT8 after max-pool and scaling")


if __name__ == "__main__":
    main()
