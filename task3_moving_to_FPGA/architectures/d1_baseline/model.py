from __future__ import annotations

import torch
import torch.nn as nn

from architectures.common.quantization import QuantSpec, qconv2d, qglobal_avg_pool, qlinear, qmaxpool2d


class BaselineCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(inputs), 1))

    def forward_int8(self, inputs: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        outputs = qconv2d(inputs, self.features[0], spec)
        outputs = qmaxpool2d(outputs)
        outputs = qconv2d(outputs, self.features[3], spec)
        outputs = qmaxpool2d(outputs)
        outputs = qconv2d(outputs, self.features[6], spec)
        outputs = qconv2d(outputs, self.features[8], spec)
        outputs = qglobal_avg_pool(outputs)
        return qlinear(torch.flatten(outputs, 1), self.classifier, spec)

    def rtl_export_layers(self):
        return [
            {"name": "conv1", "kind": "conv2d", "module": self.features[0]},
            {"name": "conv2", "kind": "conv2d", "module": self.features[3]},
            {"name": "conv3", "kind": "conv2d", "module": self.features[6]},
            {"name": "conv4", "kind": "conv2d", "module": self.features[8]},
            {"name": "linear", "kind": "linear", "module": self.classifier},
        ]


def build_model() -> BaselineCNN:
    return BaselineCNN()
