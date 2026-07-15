import argparse
import logging
import os
import warnings
from types import SimpleNamespace

import torch
import torch.nn as nn

from data_loader import get_cifar10_loaders
from model import GoldenCNNModel
import copy

from quant import (
    FakeQuantizeWeight,
    QuantBits,
    compute_fixed_point_scale,
    convert_qat,
    export_hex,
    export_requant_hex,
    fold_bn_pair,
    prepare_qat,
    quantize_bias,
)
from test import test
from train import train
from utils.config import load_config

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")


def build_bn_folded_export_model(net):
    """
    Deep-copy net and fold each conv_layers[i]/bn_layers[i] pair into a
    single bias-carrying conv. Does NOT touch the live training model —
    BN running stats are still needed if you train further or re-evaluate
    the FP32/QAT-with-BN accuracy afterward.
    """
    export_net = copy.deepcopy(net)
    if not export_net.use_bn:
        return export_net

    for i in range(export_net.num_active_layers):
        fq_weight_module = export_net.conv_layers[i]  # FakeQuantizeWeight
        bn_module = export_net.bn_layers[i]
        folded_conv = fold_bn_pair(fq_weight_module.module, bn_module)
        fq_weight_module.module = folded_conv
        export_net.bn_layers[i] = nn.Identity()
    return export_net


def build_requant_entries(export_net, frac_bits: int = 16):
    """
    Assemble the per-layer fixed-point requant table.
    s_x for block i is: input_quant's scale (i==0) or act_quant[i-1]'s scale.
    s_y for block i is: act_quant[i]'s scale (this block's own output).
    """
    entries = []
    for i in range(export_net.num_active_layers):
        fq_weight_module = export_net.conv_layers[i]
        s_x = (
            export_net.input_quant.get_scale()
            if i == 0
            else export_net.act_quant[i - 1].get_scale()
        )
        s_y = export_net.act_quant[i].get_scale()

        _, s_w = fq_weight_module.get_quantized_weights()  # per-channel scale
        M0, fb = compute_fixed_point_scale(s_x, s_w, s_y, frac_bits=frac_bits)

        bias = fq_weight_module.module.bias
        bias_int32 = (
            quantize_bias(bias.data, s_x, s_w)
            if bias is not None
            else torch.zeros_like(s_w, dtype=torch.int32)
        )

        entries.append(
            {
                "name": f"conv_layers.{i}",
                "M0": M0,
                "frac_bits": fb,
                "bias_int32": bias_int32,
            }
        )
    return entries


def save_experiment_artifacts(net, quant_bits: QuantBits):
    """Save all artifacts. Order matters — export before convert_qat."""
    print("\n--- Saving Experiment Artifacts ---")
    os.makedirs(f"models/{quant_bits.name}", exist_ok=True)

    # 1. QAT checkpoint (FakeQuantizeWeight wrappers still intact)
    qat_path = f"models/{quant_bits.name}/golden_model_qat_{quant_bits.name}.pt"
    torch.save(net.state_dict(), qat_path)
    print(f" -> Saved QAT Checkpoint:         {qat_path}")

    # 2. Scales (requires FakeQuantizeWeight to exist)
    scales = {}
    for name, module in net.named_modules():
        if isinstance(module, FakeQuantizeWeight):
            _, scale = module.get_quantized_weights()
            s = scale.detach().cpu().flatten().tolist()
            scales[name] = s if isinstance(s, list) else [s]
    scales_path = f"models/{quant_bits.name}/scales_{quant_bits.name}.pt"
    torch.save(scales, scales_path)
    print(f" -> Saved Scales Profile:         {scales_path}")

    if quant_bits != QuantBits.FL32:
        # 2b. BN-folded export copy — hardware has no BN, so weights/hex/requant
        # must come from the folded model, never the live training model.
        export_net = build_bn_folded_export_model(net)

        # 3. Hex export (requires FakeQuantizeWeight to exist)
        hex_path = f"models/{quant_bits.name}/weights_{quant_bits.name}.hex"
        export_hex(export_net, hex_path)
        print(f" -> Exported Hardware Hex:        {hex_path}")

        # 3b. Requant table: per-channel M0/frac_bits + int32 bias, derived from
        # the calibrated activation scales (input_quant / act_quant EMAs).
        requant_path = f"models/{quant_bits.name}/requant_{quant_bits.name}.hex"
        requant_entries = build_requant_entries(export_net)
        export_requant_hex(requant_entries, requant_path)
        print(f" -> Exported Requant Table:       {requant_path}")

    # 4. convert_qat LAST — strips FakeQuantizeWeight, bakes quantized weights in
    # (applied to the ORIGINAL net, since that one still has live BN layers
    # and is what you'd keep training/evaluating in PyTorch)
    convert_qat(net)
    hw_path = f"models/{quant_bits.name}/golden_model_hardware_{quant_bits.name}.pt"
    torch.save(net.state_dict(), hw_path)
    print(f" -> Saved Hardware Integer Weights: {hw_path}")


def main(force_train: bool = False, QUANT_BITS: QuantBits = QuantBits.INT4):
    config: SimpleNamespace = load_config()  # type: ignore
    model_config = config.model
    train_config = config.train

    MODEL_PATH = f"./models/{QUANT_BITS.name}/golden_model_qat_{QUANT_BITS.name}.pt"

    print(f"Pipeline Setup: [Mode: QAT] [Target Precision: {QUANT_BITS.name}]")
    print(
        f"Config: layers={model_config.layer_depth}  "
        f"channels={model_config.channels}  "
        f"use_gap={getattr(model_config, 'use_gap', True)}  "
        f"use_bn={getattr(model_config, 'use_bn', True)}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    trainloader, valloader, testloader = get_cifar10_loaders(
        root=config.data.root,
        batch_size=train_config.batch_size,
        greyscale=model_config.greyscale,
        num_workers=train_config.num_workers,
        val_split=train_config.val_split,
    )

    net = GoldenCNNModel(config=config)
    net.describe()

    # prepare_qat wraps Conv2d/Linear with FakeQuantizeWeight
    # Must be done BEFORE loading a QAT checkpoint (key names must match)
    if QUANT_BITS != QuantBits.FL32:
        prepare_qat(net, bits=QUANT_BITS)

    if os.path.exists(MODEL_PATH) and not force_train:
        print(f"\nFound checkpoint: {MODEL_PATH} — loading...")
        net.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        print("Loaded.")
    else:
        print("\nNo checkpoint found — training from scratch...")
        train(trainloader, valloader, net, device)
        print("Training complete.")
        best_ckpt = "./models/best_checkpoint.pt"
        if os.path.exists(best_ckpt):
            print("Loading best checkpoint from training...")
            net.load_state_dict(
                torch.load(best_ckpt, map_location=device, weights_only=True)
            )

    net.to(device)

    # Evaluate: float32 baseline vs quantized
    print("\n--- Accuracy Evaluation ---")
    print("Float32 (quantization OFF):")
    fl32_acc = test(testloader, classes, net, device, quantized=False)

    if QUANT_BITS != QuantBits.FL32:
        print(f"\n{QUANT_BITS.name} (quantization ON):")
        quant_acc = test(testloader, classes, net, device, quantized=True)
        print(f"\nDelta: {quant_acc - fl32_acc:+.2f}%")

    save_experiment_artifacts(net, QUANT_BITS)
    print("\nPipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate, and export the QAT CNN.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train from scratch even if a QAT checkpoint already exists.",
    )
    parser.add_argument(
        "--quant-bits",
        choices=[bits.name for bits in QuantBits],
        default=QuantBits.INT4.name,
        help="Target weight precision (default: INT4).",
    )
    args = parser.parse_args()
    main(force_train=args.train, QUANT_BITS=QuantBits[args.quant_bits])
