from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.common.quantization import (
    QuantSpec,
    qconv2d,
    qglobal_avg_pool,
    qlinear,
    qresidual_relu,
)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        main = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        main = self.bn2(self.conv2(main))
        skip = inputs if self.shortcut_conv is None else self.shortcut_bn(self.shortcut_conv(inputs))
        return F.relu(main + skip, inplace=True)

    def forward_int8(self, inputs: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        main = qconv2d(inputs, self.conv1, spec, bn=self.bn1, relu=True)
        main = qconv2d(main, self.conv2, spec, bn=self.bn2, relu=False)
        if self.shortcut_conv is None:
            skip = inputs
        else:
            skip = qconv2d(inputs, self.shortcut_conv, spec, bn=self.shortcut_bn, relu=False)
        return qresidual_relu(main, skip, spec)


class ResNet20(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, 16, blocks=3, first_stride=1)
        self.stage2 = self._make_stage(16, 32, blocks=3, first_stride=2)
        self.stage3 = self._make_stage(32, 64, blocks=3, first_stride=2)
        self.classifier = nn.Linear(64, 10)
        self._initialize()

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, blocks: int, first_stride: int) -> nn.ModuleList:
        layers = [BasicBlock(in_channels, out_channels, first_stride)]
        layers.extend(BasicBlock(out_channels, out_channels, 1) for _ in range(1, blocks))
        return nn.ModuleList(layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.relu(self.stem_bn(self.stem_conv(inputs)), inplace=True)
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                outputs = block(outputs)
        outputs = F.adaptive_avg_pool2d(outputs, (1, 1))
        return self.classifier(torch.flatten(outputs, 1))

    def forward_int8(self, inputs: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        outputs = qconv2d(inputs, self.stem_conv, spec, bn=self.stem_bn, relu=True)
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                outputs = block.forward_int8(outputs, spec)
        outputs = qglobal_avg_pool(outputs)
        return qlinear(torch.flatten(outputs, 1), self.classifier, spec)

    def rtl_export_layers(self):
        layers = [{"name": "stem", "kind": "conv2d", "module": self.stem_conv, "bn": self.stem_bn}]
        for stage_index, stage in enumerate((self.stage1, self.stage2, self.stage3), start=1):
            for block_index, block in enumerate(stage, start=1):
                prefix = f"s{stage_index}b{block_index}"
                layers.append({"name": f"{prefix}_c1", "kind": "conv2d", "module": block.conv1, "bn": block.bn1})
                layers.append({"name": f"{prefix}_c2", "kind": "conv2d", "module": block.conv2, "bn": block.bn2})
                if block.shortcut_conv is not None:
                    layers.append(
                        {
                            "name": f"{prefix}_skip",
                            "kind": "shortcut_conv2d",
                            "module": block.shortcut_conv,
                            "bn": block.shortcut_bn,
                        }
                    )
        layers.append({"name": "linear", "kind": "linear", "module": self.classifier})
        return layers


def build_model() -> ResNet20:
    return ResNet20()
