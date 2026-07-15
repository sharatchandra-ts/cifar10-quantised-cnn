import torch
import torch.nn as nn

from quant import FakeQuantizeActivation, QuantBits

NUM_CLASSES: int = 10
MAXPOOL_SIZE: int = 2


class GoldenCNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config.model

        assert 1 <= cfg.layer_depth <= len(cfg.channels), (
            f"layer_depth must be 1..{len(cfg.channels)}"
        )

        self.num_active_layers = cfg.layer_depth
        self.input_size = cfg.input_size
        self.use_gap = getattr(cfg, "use_gap", True)
        self.use_bn = getattr(cfg, "use_bn", True)

        in_ch = 1 if cfg.greyscale else 3

        # ── Conv blocks ──────────────────────────────────────────────────
        # Each block: Conv(3×3, pad=1, bias=False) → BN → LeakyReLU → MaxPool
        #   padding=1  keeps spatial size constant through conv
        #   only MaxPool halves: 32→16→8→4  (always powers of 2)
        #   bias=False  because BN absorbs bias — no separate bias needed
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for out_ch in cfg.channels:
            self.conv_layers.append(
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=cfg.kernel_size, padding=1, bias=False
                )  # BN handles bias
            )
            self.bn_layers.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch

        self.pool = nn.MaxPool2d(MAXPOOL_SIZE, MAXPOOL_SIZE)
        self.act = nn.LeakyReLU(
            negative_slope=cfg.negative_slope
        )  # avoids dead neurons vs ReLU
        self.dropout = nn.Dropout(p=cfg.dropout)

        # ── Activation quantization ─────────────────────────────────────
        # Hardware boundary: every activation tensor crossing a layer edge
        # is int8, PER-TENSOR (one scale, not per-channel like weights).
        # input_quant: quantizes the input image once, defines s_x for layer 0.
        # act_quant[i]: quantizes the output of block i (post pool), which
        #   becomes s_x for block i+1.
        self.input_quant = FakeQuantizeActivation(bits=QuantBits.INT8)
        self.act_quant = nn.ModuleList(
            [FakeQuantizeActivation(bits=QuantBits.INT8) for _ in cfg.channels]
        )
        # Quantizes the GAP output before the classifier's Linear layer,
        # since that Linear is also int4-weight and needs an int8 input.
        self.gap_quant = FakeQuantizeActivation(bits=QuantBits.INT8)

        # ── Classifier ───────────────────────────────────────────────────
        final_ch = cfg.channels[self.num_active_layers - 1]

        if self.use_gap:
            # Global Average Pool: [B, C, H, W] → [B, C]
            # Replaces flatten + large FC stack — cuts params dramatically
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc_layers = nn.ModuleList()  # empty, kept for compat
            self.classifier = nn.Linear(final_ch, NUM_CLASSES)
        else:
            flat_size = self._compute_flat_size(1 if cfg.greyscale else 3)
            fc_in = flat_size
            self.fc_layers = nn.ModuleList()
            self.fc_quant = nn.ModuleList()
            for fc_out in cfg.fc_sizes:
                self.fc_layers.append(nn.Linear(fc_in, fc_out))
                self.fc_quant.append(FakeQuantizeActivation(bits=QuantBits.INT8))
                fc_in = fc_out
            self.classifier = nn.Linear(fc_in, NUM_CLASSES)

    def _compute_flat_size(self, in_ch: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, in_ch, self.input_size, self.input_size)
            for i in range(self.num_active_layers):
                x = self.pool(self.act(self.bn_layers[i](self.conv_layers[i](x))))
            return x.numel()

    def forward(self, x):
        # Quantize the input image -- defines s_x for the very first conv.
        # (If your dataloader already emits int8-range values this just
        # re-affirms that scale; if it normalizes to e.g. [-1, 1] this is
        # where that gets mapped into the int8 domain the hardware expects.)
        x = self.input_quant(x)

        for i in range(self.num_active_layers):
            x = self.conv_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.act(x)
            x = self.pool(x)
            # Hardware layer boundary: requantize back to int8 before the
            # next conv reads it. This output scale IS s_x for block i+1.
            x = self.act_quant[i](x)

        if self.use_gap:
            x = self.gap(x)  # [B, C, H, W] → [B, C, 1, 1]
            x = x.flatten(1)  # [B, C, 1, 1] → [B, C]
            x = self.gap_quant(x)  # int8 input for the classifier Linear
        else:
            x = torch.flatten(x, 1)
            for fc, fc_q in zip(self.fc_layers, self.fc_quant):
                x = fc_q(self.act(fc(x)))

        x = self.dropout(x)
        return self.classifier(x)

    def describe(self):
        in_ch = 1
        spatial = self.input_size
        total = 0
        print(f"\n{'─' * 58}")
        print(f"  {'Layer':<28} {'Output shape':<18} {'Params':>8}")
        print(f"{'─' * 58}")
        print(f"  {'Input':<28} {str([in_ch, spatial, spatial]):<18}")

        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            spatial = spatial // 2
            tag = "▶ ACTIVE" if i < self.num_active_layers else "  inactive"
            w = conv.weight.numel()
            b = bn.weight.numel() * 2 if self.use_bn else 0
            total += w + b
            shape = [conv.out_channels, spatial, spatial]
            print(f"  {tag} Conv{i + 1}+BN+Act+Pool  {str(shape):<18} {w + b:>8,}")

        final_ch = self.conv_layers[self.num_active_layers - 1].out_channels
        if self.use_gap:
            c = self.classifier.weight.numel() + self.classifier.bias.numel()
            total += c
            print(f"  {'GAP':<28} {str([final_ch]):<18} {'—':>8}")
            print(f"  {'Classifier (Linear)':<28} {str([NUM_CLASSES]):<18} {c:>8,}")
        else:
            for fc in self.fc_layers:
                p = fc.weight.numel() + fc.bias.numel()
                total += p
                print(f"  {'FC':<28} {str([fc.out_features]):<18} {p:>8,}")
            c = self.classifier.weight.numel() + self.classifier.bias.numel()
            total += c
            print(f"  {'Classifier':<28} {str([NUM_CLASSES]):<18} {c:>8,}")

        print(f"{'─' * 58}")
        print(f"  {'TOTAL':<28} {'':18} {total:>8,}")
        print(f"{'─' * 58}\n")
