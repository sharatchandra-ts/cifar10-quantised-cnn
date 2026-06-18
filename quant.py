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
    """Enable/disable fake quantization independently of train/eval mode."""
    for module in model.modules():
        if isinstance(module, FakeQuantizeWeight):
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
