from datetime import datetime
import json
import os

import torch

from quant import FakeQuantizeWeight, QuantBits

def save_stats_md(
    quant_bits: QuantBits,
    net: torch.nn.Module,
    train_time_s: float | None,
    avg_latency_ms: float,
    fl32_acc: float,
    quant_acc: float,
    config,
):
    os.makedirs('models', exist_ok=True)
    stats_path = 'models/stats.md'
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_params = sum(p.numel() for p in net.parameters())

    lines = []
    lines.append(f"\n---\n")
    lines.append(f"## Run — {quant_bits.name} | {now}\n")

    # Model
    lines.append(f"### Model\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Active layers | {config.model.layer_depth} |")
    lines.append(f"| Channels | {config.model.channels} |")
    lines.append(f"| FC sizes | {config.model.fc_sizes} |")
    lines.append(f"| Greyscale | {config.model.greyscale} |")
    lines.append(f"| Total params | {total_params:,} |")
    lines.append(f"| Float32 size | {total_params * 4 / 1024:.2f} KB |")
    lines.append(f"| {quant_bits.name} size | {total_params * quant_bits.value / 8 / 1024:.2f} KB |")

    # Training
    lines.append(f"\n### Training\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Epochs | {config.train.epochs} |")
    lines.append(f"| Learning rate | {config.train.learning_rate} |")
    lines.append(f"| Batch size | {config.train.batch_size} |")
    lines.append(f"| Train time | {f'{train_time_s:.1f}s' if train_time_s else 'loaded from checkpoint'} |")

    # Accuracy
    delta = quant_acc - fl32_acc
    delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
    lines.append(f"\n### Accuracy\n")
    lines.append(f"| Mode | Accuracy |")
    lines.append(f"|---|---|")
    lines.append(f"| Float32 | {fl32_acc:.2f}% |")
    lines.append(f"| {quant_bits.name} | {quant_acc:.2f}% |")
    lines.append(f"| Delta | {delta_str} |")
    lines.append(f"| Avg latency | {avg_latency_ms:.3f} ms/batch |")

    # Per-layer quant stats
    lines.append(f"\n### Layer Statistics\n")
    lines.append(f"| Layer | Shape | Weights | Bits | Scale min | Scale max | W min | W max | W mean | W std | Range util |")
    lines.append(f"|---|---|---|---|---|---|---|---|---|---|---|")

    for name, module in net.named_modules():
        if not isinstance(module, FakeQuantizeWeight):
            continue
        w_int, scale = module.get_quantized_weights()
        w_flat = w_int.float().flatten()
        s = scale.detach().cpu().flatten()
        util = 100 * (w_flat.max() - w_flat.min()).item() / (module.q_max - module.q_min)
        lines.append(
            f"| {name} "
            f"| {list(w_int.shape)} "
            f"| {w_int.numel():,} "
            f"| INT{module.bits.value} "
            f"| {s.min():.6f} "
            f"| {s.max():.6f} "
            f"| {w_flat.min():.0f} "
            f"| {w_flat.max():.0f} "
            f"| {w_flat.mean():.3f} "
            f"| {w_flat.std():.3f} "
            f"| {util:.1f}% |"
        )

    # Scales
    lines.append(f"\n### Scales (per output channel)\n")
    for name, module in net.named_modules():
        if not isinstance(module, FakeQuantizeWeight):
            continue
        _, scale = module.get_quantized_weights()
        s = [f'{v:.6f}' for v in scale.detach().cpu().flatten().tolist()]
        lines.append(f"**{name}**: `{s}`\n")

    # Write — append so history is preserved
    with open(stats_path, 'a') as f:
        f.write('\n'.join(lines))
        f.write('\n')

    print(f"  Stats appended to {stats_path}")
    

def save_stats(
    quant_bits: QuantBits,
    net: torch.nn.Module,
    train_time_s: float | None,
    avg_latency_ms: float,
    fl32_acc: float,
    quant_acc: float,
    config,
):
    """Append stats for this run to a JSON file for documentation."""
    os.makedirs('models', exist_ok=True)
    stats_path = 'models/stats.json'

    # Load existing stats if file exists
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            all_stats = json.load(f)
    else:
        all_stats = []

    # Collect per-layer quant info
    layer_info = []
    for name, module in net.named_modules():
        if isinstance(module, FakeQuantizeWeight):
            w_int, scale = module.get_quantized_weights()
            w_flat = w_int.float().flatten()
            s = scale.detach().cpu().flatten().tolist()
            layer_info.append({
                'name': name,
                'shape': list(w_int.shape),
                'num_weights': w_int.numel(),
                'bits': module.bits.value,
                'scale': s if isinstance(s, list) else [s],
                'w_min': w_flat.min().item(),
                'w_max': w_flat.max().item(),
                'w_mean': round(w_flat.mean().item(), 4),
                'w_std': round(w_flat.std().item(), 4),
                'range_utilisation_pct': round(
                    100 * (w_flat.max() - w_flat.min()).item()
                    / (module.q_max - module.q_min), 2
                ),
            })

    total_params = sum(p.numel() for p in net.parameters())

    entry = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'quant': quant_bits.name,
        'model': {
            'total_params': total_params,
            'float32_size_kb': round(total_params * 4 / 1024, 2),
            'quantized_size_kb': round(total_params * quant_bits.value / 8 / 1024, 2),
            'active_layers': config.model.layer_depth,
            'channels': config.model.channels,
            'fc_sizes': config.model.fc_sizes,
            'greyscale': config.model.greyscale,
        },
        'train': {
            'epochs': config.train.epochs,
            'learning_rate': config.train.learning_rate,
            'batch_size': config.train.batch_size,
            'train_time_s': round(train_time_s, 1) if train_time_s else None,
        },
        'accuracy': {
            'float32_pct': round(fl32_acc, 2),
            'quantized_pct': round(quant_acc, 2),
            'delta_pct': round(quant_acc - fl32_acc, 2),
        },
        'inference': {
            'avg_latency_ms': round(avg_latency_ms, 3),
        },
        'layers': layer_info,
    }

    all_stats.append(entry)

    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n  Stats appended to {stats_path}")