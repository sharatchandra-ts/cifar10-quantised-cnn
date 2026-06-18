import logging
import os
import warnings

import torch

from data_loader import get_cifar10_loaders
from model import GoldenCNNModel
from quant import FakeQuantizeWeight, QuantBits, convert_qat, export_hex, prepare_qat
from test import test
from train import train
from utils.config import load_config

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")


def save_experiment_artifacts(net, quant_bits: QuantBits):
    """Save all artifacts. Order matters — export before convert_qat."""
    print("\n--- Saving Experiment Artifacts ---")
    os.makedirs("models", exist_ok=True)

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

    # 3. Hex export (requires FakeQuantizeWeight to exist)
    hex_path = f"models/{quant_bits.name}/weights_{quant_bits.name}.hex"
    export_hex(net, hex_path)
    print(f" -> Exported Hardware Hex:        {hex_path}")

    # 4. convert_qat LAST — strips FakeQuantizeWeight, bakes quantized weights in
    convert_qat(net)
    hw_path = f"models/{quant_bits.name}/golden_model_hardware_{quant_bits.name}.pt"
    torch.save(net.state_dict(), hw_path)
    print(f" -> Saved Hardware Integer Weights: {hw_path}")


def main(force_train: bool = False, QUANT_BITS: QuantBits = QuantBits.INT4):
    config = load_config()
    MODEL_PATH = f"./models/{QUANT_BITS.name}/golden_model_qat_{QUANT_BITS.name}.pt"

    print(f"Pipeline Setup: [Mode: QAT] [Target Precision: {QUANT_BITS.name}]")
    print(
        f"Config: layers={config.model.layer_depth}  "
        f"channels={config.model.channels}  "
        f"use_gap={getattr(config.model, 'use_gap', True)}  "
        f"use_bn={getattr(config.model, 'use_bn', True)}"
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
        root="./data",
        batch_size=config.train.batch_size,
        greyscale=config.model.greyscale,
        num_workers=config.train.num_workers,
        val_split=config.train.val_split,
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
    main(force_train=True, QUANT_BITS=QuantBits.INT4)
