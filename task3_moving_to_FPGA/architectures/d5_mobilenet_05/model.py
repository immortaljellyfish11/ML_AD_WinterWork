from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.common.quantization import QuantSpec, qconv2d, qglobal_avg_pool, qlinear


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=kernel // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(inputs)), inplace=True)

    def forward_int8(self, inputs: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        return qconv2d(inputs, self.conv, spec, bn=self.bn, relu=True)


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = ConvBNReLU(in_channels, in_channels, 3, stride, groups=in_channels)
        self.pointwise = ConvBNReLU(in_channels, out_channels, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(inputs))

    def forward_int8(self, inputs: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        return self.pointwise.forward_int8(self.depthwise.forward_int8(inputs, spec), spec)


class MobileNetV1Half(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNReLU(3, 16, 3, 1)
        settings = [
            (32, 1),
            (64, 2),
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (256, 1),
            (512, 2),
            (512, 1),
        ]
        blocks = []
        in_channels = 16
        for out_channels, stride in settings:
            blocks.append(DepthwiseSeparable(in_channels, out_channels, stride))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(512, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        for block in self.blocks:
            outputs = block(outputs)
        outputs = F.adaptive_avg_pool2d(outputs, (1, 1))
        return self.classifier(torch.flatten(outputs, 1))

    def forward_int8(self, inputs: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        outputs = self.stem.forward_int8(inputs, spec)
        for block in self.blocks:
            outputs = block.forward_int8(outputs, spec)
        outputs = qglobal_avg_pool(outputs)
        return qlinear(torch.flatten(outputs, 1), self.classifier, spec)

    def rtl_export_layers(self):
        layers = [{"name": "stem", "kind": "conv2d", "module": self.stem.conv, "bn": self.stem.bn}]
        for index, block in enumerate(self.blocks, start=1):
            layers.append(
                {
                    "name": f"block{index}_depthwise",
                    "kind": "depthwise_conv2d",
                    "module": block.depthwise.conv,
                    "bn": block.depthwise.bn,
                }
            )
            layers.append(
                {
                    "name": f"block{index}_pointwise",
                    "kind": "pointwise_conv2d",
                    "module": block.pointwise.conv,
                    "bn": block.pointwise.bn,
                }
            )
        layers.append({"name": "linear", "kind": "linear", "module": self.classifier})
        return layers


def build_model() -> MobileNetV1Half:
    return MobileNetV1Half()
