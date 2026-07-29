from __future__ import annotations

import torch
import torch.nn as nn

from architectures.common.quantization import QuantSpec, qconv2d, qglobal_avg_pool, qlinear, qmaxpool2d


class SmallCNN(nn.Module):
    """D2 16-32-64 CIFAR-10 CNN / D2 16-32-64 小型 CIFAR-10 网络。"""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(inputs), 1))

    def forward_int8(self, inputs: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        """Legacy global-Q0.7 path; final evidence uses common/per_layer_ptq.py.

        历史全局 Q0.7 接口；最终量化结果统一使用 common/per_layer_ptq.py。
        """
        outputs = qmaxpool2d(qconv2d(inputs, self.features[0], spec))
        outputs = qmaxpool2d(qconv2d(outputs, self.features[3], spec))
        outputs = qglobal_avg_pool(qconv2d(outputs, self.features[6], spec))
        return qlinear(torch.flatten(outputs, 1), self.classifier, spec)

    def rtl_export_layers(self):
        """Expose ordered trainable layers to PTQ export / 向 PTQ 导出器提供有序可训练层。"""
        return [
            {"name": "conv1", "kind": "conv2d", "module": self.features[0]},
            {"name": "conv2", "kind": "conv2d", "module": self.features[3]},
            {"name": "conv3", "kind": "conv2d", "module": self.features[6]},
            {"name": "linear", "kind": "linear", "module": self.classifier},
        ]


def build_model() -> SmallCNN:
    return SmallCNN()
