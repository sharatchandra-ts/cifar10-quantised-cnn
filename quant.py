# quant.py
import os
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantBits(Enum):
    INT4 = 4
    INT8 = 8
    FL32 = 32


def _get_range(bits: QuantBits):
    b = bits.value
    return -(2 ** (b - 1)), (2 ** (b - 1)) - 1


def _per_channel_scale(w: torch.Tensor, q_max: int) -> torch.Tensor:
    """
    One scale per output channel (dim 0), works for Conv2d and Linear.
    Conv2d: [Co, Ci, Kh, Kw] -> scale shape [Co, 1, 1, 1]
    Linear: [out, in]         -> scale shape [out, 1]
    """
    scale = w.detach().cpu().flatten(1).abs().max(dim=1).values / q_max
    scale = scale.clamp(min=1e-8)
    for _ in range(w.dim() - 1):
        scale = scale.unsqueeze(-1)
    return scale


class FakeQuantizeWeight(nn.Module):
    def __init__(self, module: nn.Module, bits: QuantBits = QuantBits.INT4):
        super().__init__()
        self.module = module
        self.bits = bits
        self.q_min, self.q_max = _get_range(bits)
        self.enabled = True  # <--- Track state explicitly

    def forward(self, x):
        # Read the explicit enabled flag instead of self.training
        if self.enabled and self.bits != QuantBits.FL32:
            w_fq = self._fake_quantize(self.module.weight)
            if isinstance(self.module, nn.Conv2d):
                return F.conv2d(
                    x,
                    w_fq,
                    self.module.bias,
                    self.module.stride,
                    self.module.padding,
                    self.module.dilation,
                    self.module.groups,
                )
            else:
                return F.linear(x, w_fq, self.module.bias)
        return self.module(x)

    def _fake_quantize(self, w: torch.Tensor) -> torch.Tensor:
        scale = _per_channel_scale(w, self.q_max).to(w.device)
        w_fq = (w / scale).round().clamp(self.q_min, self.q_max) * scale
        # STE: quantized value in forward, straight-through gradient in backward
        return w + (w_fq - w).detach()

    def get_quantized_weights(self):
        """Returns (w_int, scale) on CPU. Safe for numpy/hex export."""
        w = self.module.weight.data.cpu()
        scale = _per_channel_scale(w, self.q_max)
        w_int = (w / scale).round().clamp(self.q_min, self.q_max).to(torch.int8)
        return w_int, scale.squeeze()


class FakeQuantizeActivation(nn.Module):
    """
    Per-TENSOR (not per-channel) activation fake-quantizer.

    Hardware only has one scale multiplier per activation tensor crossing a
    layer boundary — not one per channel — so this must NOT mirror the
    per-channel weight scheme in FakeQuantizeWeight.

    Calibrates its clipping range with a running max (EMA) observed during
    training, the same way TFLite's MovingAverageMinMax observer works.
    Symmetric, zero-point = 0 — matches the rest of this codebase and is
    required for ReLU/MaxPool to commute correctly with quantization.
    """

    def __init__(self, bits: QuantBits = QuantBits.INT8, momentum: float = 0.9):
        super().__init__()
        self.bits = bits
        self.q_min, self.q_max = _get_range(bits)
        self.momentum = momentum
        self.enabled = True
        self.register_buffer("running_max", torch.tensor(1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                batch_max = x.detach().abs().max()
                # EMA warm start: first observed batch sets the range directly
                # instead of slowly ramping up from the 1e-8 buffer default.
                if self.running_max.item() <= 1e-7:
                    self.running_max.copy_(batch_max)
                else:
                    self.running_max.mul_(self.momentum).add_(
                        batch_max, alpha=1 - self.momentum
                    )

        if self.enabled and self.bits != QuantBits.FL32:
            scale = (self.running_max / self.q_max).clamp(min=1e-8)
            x_fq = (x / scale).round().clamp(self.q_min, self.q_max) * scale
            return x + (x_fq - x).detach()
        return x

    def get_scale(self) -> float:
        """Scalar float scale for this activation tensor, for export."""
        return (self.running_max / self.q_max).clamp(min=1e-8).item()


def fold_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fold BatchNorm2d into the preceding Conv2d (bias=False assumed).

    w_folded = w * gamma / sqrt(running_var + eps)
    b_folded = beta - running_mean * gamma / sqrt(running_var + eps)

    Returns a NEW Conv2d with bias=True holding the folded params. Does not
    mutate the input modules — call this on a deep-copied model intended
    only for export, since folding is a one-way, inference-only transform
    (BN running stats are still needed for further training).
    """
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )

    with torch.no_grad():
        std = torch.sqrt(bn.running_var + bn.eps)
        gamma_over_std = bn.weight / std  # shape [Co]

        fused.weight.copy_(
            conv.weight * gamma_over_std.reshape(-1, 1, 1, 1)
        )
        fused.bias.copy_(bn.bias - bn.running_mean * gamma_over_std)

    return fused


def compute_fixed_point_scale(
    s_x: float, s_w, s_y: float, frac_bits: int = 16
):
    """
    Compute the per-output-channel fixed-point requant multiplier.

    Real-valued M = (s_x * s_w) / s_y  is decomposed as  M ~= M0 / 2^frac_bits
    so hardware does  y = (acc * M0) >> frac_bits  -- integer multiply +
    shift, no float divide anywhere in the datapath.

    s_w may be a scalar or a per-channel tensor (matches FakeQuantizeWeight's
    per-channel scale) -- M0 comes out with the same shape.

    Returns (M0: int16 tensor, frac_bits: int).
    """
    s_w_t = s_w if torch.is_tensor(s_w) else torch.tensor(float(s_w))
    M = (s_x * s_w_t) / s_y
    M0 = (M * (2 ** frac_bits)).round().clamp(-(2 ** 15), 2 ** 15 - 1).to(torch.int16)
    return M0, frac_bits


def prepare_qat(model: nn.Module, bits: QuantBits = QuantBits.INT4) -> nn.Module:
    """Wrap all Conv2d and Linear layers with FakeQuantizeWeight."""
    for name, module in list(model.named_children()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            setattr(model, name, FakeQuantizeWeight(module, bits))
        elif isinstance(module, nn.ModuleList):
            # setattr doesn't work on ModuleList — must use index assignment
            for i, submodule in enumerate(module):
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    module[i] = FakeQuantizeWeight(submodule, bits)
                else:
                    prepare_qat(submodule, bits)
        else:
            prepare_qat(module, bits)
    return model


def convert_qat(model: nn.Module) -> nn.Module:
    """Unwrap FakeQuantizeWeight, baking quantized weights back in as float."""
    for name, module in list(model.named_children()):
        if isinstance(module, FakeQuantizeWeight):
            w_int, scale = module.get_quantized_weights()
            # Broadcast scale back to weight shape
            scale_bc = scale
            for _ in range(module.module.weight.dim() - 1):
                scale_bc = scale_bc.unsqueeze(-1)
            module.module.weight.data.copy_(
                w_int.float().to(module.module.weight.device)
                * scale_bc.to(module.module.weight.device)
            )
            setattr(model, name, module.module)
        elif isinstance(module, nn.ModuleList):
            for i, submodule in enumerate(module):
                if isinstance(submodule, FakeQuantizeWeight):
                    w_int, scale = submodule.get_quantized_weights()
                    scale_bc = scale
                    for _ in range(submodule.module.weight.dim() - 1):
                        scale_bc = scale_bc.unsqueeze(-1)
                    submodule.module.weight.data.copy_(
                        w_int.float().to(submodule.module.weight.device)
                        * scale_bc.to(submodule.module.weight.device)
                    )
                    module[i] = submodule.module
                else:
                    convert_qat(submodule)
        else:
            convert_qat(module)
    return model


def set_fake_quant(model: nn.Module, enabled: bool):
    """Enable/disable fake quantization (weights AND activations) independently
    of train/eval mode."""
    for module in model.modules():
        if isinstance(module, (FakeQuantizeWeight, FakeQuantizeActivation)):
            module.enabled = enabled


def export_hex(model: nn.Module, path: str = "./models/weights.hex"):
    """
    Export quantized weights to $readmemh-compatible hex file.
    Must be called BEFORE convert_qat.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for name, module in model.named_modules():
            if not isinstance(module, FakeQuantizeWeight):
                continue

            w_int, scale = module.get_quantized_weights()  # always CPU
            bits = module.bits.value

            f.write(f"// {name}  shape={list(w_int.shape)}  bits={bits}\n")
            f.write(
                f"// scale per output channel: {scale.detach().flatten().tolist()}\n"
            )

            # view as uint8 for correct two's complement representation
            # int8 -8 = 0xF8, not 248 — view() preserves bit pattern, astype() doesn't
            w_bytes = w_int.numpy().flatten().view(np.uint8)
            for val in w_bytes:
                f.write(f"{val:02x}\n")
            f.write("\n")

    print(f"Weights exported to {path}")


def export_requant_hex(entries: list, path: str = "./models/requant.hex"):
    """
    Write the per-layer, per-channel fixed-point requant table.

    entries: list of dicts, one per conv/linear layer, each with:
        {"name": str, "M0": torch.Tensor (int16, per-channel), "frac_bits": int,
         "bias_int32": torch.Tensor (optional, int32 per-channel)}

    bias_int32 is quantized in the ACCUMULATOR's domain (scale = s_x * s_w,
    same as the raw MAC output) so it can be added directly to the accumulator
    before the M0 multiply -- NOT in the int8 activation domain.

    Written alongside weights_*.hex — weights.hex has the int4 weight bytes,
    requant.hex has the int16 multiplier constants and int32 bias the PCPI
    FSM needs to do:  y_int8 = requant((acc + bias_int32) * M0, frac_bits)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            name = entry["name"]
            M0 = entry["M0"]
            frac_bits = entry["frac_bits"]
            f.write(f"// {name}  frac_bits={frac_bits}  channels={M0.numel()}\n")
            # int16 -> uint16 two's complement
            m0_np = M0.numpy().astype(np.int16).view(np.uint16)
            for val in m0_np:
                f.write(f"m0 {val:04x}\n")

            if "bias_int32" in entry:
                bias_np = entry["bias_int32"].numpy().astype(np.int32).view(np.uint32)
                for val in bias_np:
                    f.write(f"b  {val:08x}\n")
            f.write("\n")
    print(f"Requant table exported to {path}")


def quantize_bias(bias: torch.Tensor, s_x: float, s_w: torch.Tensor) -> torch.Tensor:
    """
    Quantize a folded-BN bias into the MAC accumulator's fixed-point domain.
    scale = s_x * s_w (per output channel) -- standard for quantized conv,
    since the raw accumulator is already implicitly in units of s_x * s_w
    before any requant multiply is applied.
    """
    scale = (s_x * s_w).clamp(min=1e-12)
    return (bias.cpu() / scale.flatten()).round().to(torch.int32)


def export_weights(model: nn.Module):
    """Yield (name, w_int, scale) for each quantized layer."""
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizeWeight):
            w_int, scale = module.get_quantized_weights()
            print(
                f"{name}: shape={list(w_int.shape)}, "
                f"bits={module.bits.value}, "
                f"scale_range=[{scale.flatten().min():.6f}, {scale.flatten().max():.6f}]"
            )
            yield name, w_int, scale
