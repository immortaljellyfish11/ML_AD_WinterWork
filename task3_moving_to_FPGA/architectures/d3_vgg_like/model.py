from __future__ import annotations

import torch
import torch.nn as nn

from architectures.common.quantization import QuantSpec, qconv2d, qglobal_avg_pool, qlinear, qmaxpool2d


class VGGLikeCNN(nn.Module):
    """D3 four-convolution VGG-like CNN with 16→32→64→128 channels."""
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(3, 16, 3, padding=1),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.Conv2d(64, 128, 3, padding=1),
            ]
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.relu(self.convs[0](inputs))
        outputs = torch.relu(self.convs[1](outputs))
        outputs = nn.functional.max_pool2d(outputs, 2, 2)
        outputs = torch.relu(self.convs[2](outputs))
        outputs = torch.relu(self.convs[3](outputs))
        outputs = nn.functional.max_pool2d(outputs, 2, 2)
        outputs = nn.functional.adaptive_avg_pool2d(outputs, (1, 1))
        return self.classifier(torch.flatten(outputs, 1))

    def forward_int8(self, inputs: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        """Legacy global-Q0.7 path; final evidence uses common/per_layer_ptq.py.

        历史全局 Q0.7 接口；最终量化结果统一使用 common/per_layer_ptq.py。
        """
        outputs = qconv2d(inputs, self.convs[0], spec)
        outputs = qmaxpool2d(qconv2d(outputs, self.convs[1], spec))
        outputs = qconv2d(outputs, self.convs[2], spec)
        outputs = qmaxpool2d(qconv2d(outputs, self.convs[3], spec))
        outputs = qglobal_avg_pool(outputs)
        return qlinear(torch.flatten(outputs, 1), self.classifier, spec)

    def rtl_export_layers(self):
        """Expose ordered trainable layers to PTQ export / 向 PTQ 导出器提供有序可训练层。"""
        layers = [
            {"name": f"conv{index + 1}", "kind": "conv2d", "module": conv}
            for index, conv in enumerate(self.convs)
        ]
        return [*layers, {"name": "linear", "kind": "linear", "module": self.classifier}]


def build_model() -> VGGLikeCNN:
    return VGGLikeCNN()
