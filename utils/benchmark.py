import logging
import os
import time
import warnings

import torch

from data_loader import get_cifar10_loaders
from model import GoldenCNNModel
from quant import FakeQuantizeWeight, QuantBits, prepare_qat
from test import test
from utils.config import load_config
from utils.save_stats import save_stats, save_stats_md

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_model_stats(net: torch.nn.Module, quant_bits: QuantBits):
    """Print parameter counts, memory footprint, and quantization info."""
    print_section("MODEL STATISTICS")

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"  Total parameters:     {total_params:>10,}")
    print(f"  Trainable parameters: {trainable_params:>10,}")
    print(f"  Float32 size:         {total_params * 4 / 1024:>10.2f} KB")
    print(
        f"  {quant_bits.name} size:          "
        f"{total_params * quant_bits.value / 8 / 1024:>10.2f} KB"
    )

    print(f"\n  {'Layer':<45} {'Shape':<25} {'Params':>8}  {'Bits'}")
    print(f"  {'-' * 45} {'-' * 25} {'-' * 8}  {'-' * 4}")
    for name, module in net.named_modules():
        if isinstance(module, FakeQuantizeWeight):
            w = module.module.weight
            print(
                f"  {name:<45} {str(list(w.shape)):<25} "
                f"{w.numel():>8,}  {module.bits.value}"
            )
        elif hasattr(module, "weight") and not any(
            isinstance(p, FakeQuantizeWeight)
            for p in [net] + list(net.modules())
            if hasattr(p, "module") and p.module is module
        ):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                w = module.weight
                print(f"  {name:<45} {str(list(w.shape)):<25} {w.numel():>8,}  fp32")


def print_quantization_stats(net: torch.nn.Module):
    """Print per-layer quantization scale and weight distribution."""
    print_section("QUANTIZATION STATISTICS")

    has_quant_layers = False
    for name, module in net.named_modules():
        if not isinstance(module, FakeQuantizeWeight):
            continue

        has_quant_layers = True
        w_int, scale = module.get_quantized_weights()
        w_flat = w_int.float().flatten()

        print(f"\n  [{name}]  bits={module.bits.value}  shape={list(w_int.shape)}")
        print(
            f"    Scale:   min={scale.min():.6f}  "
            f"max={scale.max():.6f}  mean={scale.mean():.6f}"
        )
        print(
            f"    Weights: min={w_flat.min():.0f}  "
            f"max={w_flat.max():.0f}  "
            f"mean={w_flat.mean():.3f}  "
            f"std={w_flat.std():.3f}"
        )

        q_range = module.q_max - module.q_min
        actual_range = w_flat.max() - w_flat.min()
        utilisation = 100 * actual_range / q_range
        print(
            f"    Range utilisation: {utilisation:.1f}%  "
            f"({actual_range:.0f} / {q_range} levels used)"
        )

        counts = torch.histc(w_flat, bins=8, min=module.q_min, max=module.q_max)
        total = w_flat.numel()
        print("    Distribution: ", end="")
        for c in counts:
            bar = int(20 * c.item() / total)
            print(f"{'█' * bar or '░'}", end=" ")
        print()

    if not has_quant_layers:
        print("  No quantized layers found (Pure Float32 Model).")


def benchmark_inference(
    net: torch.nn.Module, testloader, device: torch.device, n_batches: int = 100
):
    """Measure inference latency and throughput."""
    print_section("INFERENCE BENCHMARK")

    net.eval()
    net.to(device)

    # Warmup
    images, _ = next(iter(testloader))
    images = images.to(device)
    for _ in range(10):
        with torch.no_grad():
            net(images)

    batch_size = images.shape[0]
    latencies = []

    with torch.no_grad():
        for i, (images, _) in enumerate(testloader):
            if i >= n_batches:
                break
            images = images.to(device)

            start = time.perf_counter()
            net(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # ms

    latencies = torch.tensor(latencies)
    avg_ms = latencies.mean().item()
    p50_ms = latencies.median().item()
    p95_ms = latencies.kthvalue(int(0.95 * len(latencies))).values.item()
    throughput = batch_size / (avg_ms / 1000)

    print(f"  Batches measured:  {n_batches}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Avg latency:       {avg_ms:.3f} ms/batch")
    print(f"  P50 latency:       {p50_ms:.3f} ms/batch")
    print(f"  P95 latency:       {p95_ms:.3f} ms/batch")
    print(f"  Throughput:        {throughput:.1f} images/sec")
    print(f"  Per-image:         {avg_ms / batch_size:.4f} ms/image")

    return avg_ms


def main():
    t_total_start = time.perf_counter()

    config = load_config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    _, _, testloader = get_cifar10_loaders(
        root="./data",
        batch_size=config.train.batch_size,
        greyscale=config.model.greyscale,
        num_workers=config.train.num_workers,
        val_split=config.train.val_split,
    )

    # ------------------------------------------------------------------ #
    #  Test all models                                                   #
    # ------------------------------------------------------------------ #
    for QUANT_BITS in QuantBits:
        print_section(f"EVALUATION SUITE — {QUANT_BITS.name}")

        net = GoldenCNNModel(config=config)

        # Only setup QAT module wrappers if we aren't working with native Float32
        if QUANT_BITS != QuantBits.FL32:
            prepare_qat(net, bits=QUANT_BITS)
            MODEL_PATH = (
                f"./models/{QUANT_BITS.name}/golden_model_qat_{QUANT_BITS.name}.pt"
            )
        else:
            # Check for a specific vanilla float32 checkpoint if you have one,
            # otherwise defaults to base weights or falls back to standard qat file
            MODEL_PATH = (
                f"./models/{QUANT_BITS.name}/golden_model_qat_{QUANT_BITS.name}.pt"
            )
            if not os.path.exists(MODEL_PATH):
                # Fallback check for alternative standard naming conventions
                MODEL_PATH = f"./models/{QUANT_BITS.name}/golden_model.pt"

        if not os.path.exists(MODEL_PATH):
            print(f"  Skipping {QUANT_BITS.name}: Checkpoint not found at {MODEL_PATH}")
            continue

        print(f"  Loading {QUANT_BITS.name} model from {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, weights_only=True)

        # Fix key mismatches for quantized variants
        if QUANT_BITS != QuantBits.FL32:
            if any(".module." in k for k in net.state_dict().keys()) and not any(
                ".module." in k for k in state_dict.keys()
            ):
                new_sd = {}
                for k, v in state_dict.items():
                    new_k = k.replace(".weight", ".module.weight").replace(
                        ".bias", ".module.bias"
                    )
                    new_sd[new_k] = v
                state_dict = new_sd

        net.load_state_dict(state_dict)
        net.to(device)

        # Performance and Stats Reporting
        print_model_stats(net, QUANT_BITS)
        print_quantization_stats(net)

        # Accuracy evaluations
        print("\n  Running Accuracy Checks...")
        net.train()  # ensure hooks stay active if required during fake quantization test
        fl32_acc = test(testloader, classes, net, device, quantized=False)
        quant_acc = test(
            testloader, classes, net, device, quantized=(QUANT_BITS != QuantBits.FL32)
        )
        net.eval()

        # Latency Evaluation
        avg_latency = benchmark_inference(net, testloader, device)

        # Save stats to markdown and json
        print("\n  Saving statistics to files...")
        save_stats(QUANT_BITS, net, None, avg_latency, fl32_acc, quant_acc, config)
        save_stats_md(QUANT_BITS, net, None, avg_latency, fl32_acc, quant_acc, config)

    print_section("DONE")
    print(f"  Total pipeline time: {time.perf_counter() - t_total_start:.1f}s")


if __name__ == "__main__":
    main()
