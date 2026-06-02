from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# 训练脚本和定点仿真脚本必须使用完全一致的归一化参数。
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _to_numpy(array_like) -> np.ndarray:
    """把 torch Tensor 或普通数组统一转换成 NumPy 数组。"""
    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    return np.asarray(array_like)


def _clip_to_int32(values: np.ndarray) -> np.ndarray:
    """对中间结果做饱和截断，避免 int64/int32 溢出后产生回绕。"""
    return np.clip(values, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)


def quantize_to_fixed_point(array_like, frac_bits: int) -> np.ndarray:
    """把浮点数转换为 Q 格式整数，所有参数都存成 32bit 有符号整数。"""
    scale = 1 << frac_bits
    values = np.round(_to_numpy(array_like).astype(np.float64) * scale)
    return _clip_to_int32(values)


def _canonical_param_name(name: str) -> str:
    """把训练脚本里的真实参数名统一映射到定点仿真使用的规范名称。"""
    aliases = {
        "features.0.weight": "conv1_weight",
        "features.0.bias": "conv1_bias",
        "features.3.weight": "conv2_weight",
        "features.3.bias": "conv2_bias",
        "features.6.weight": "conv3_weight",
        "features.6.bias": "conv3_bias",
        "classifier.weight": "classifier_weight",
        "classifier.bias": "classifier_bias",
        "conv1.weight": "conv1_weight",
        "conv1.bias": "conv1_bias",
        "conv2.weight": "conv2_weight",
        "conv2.bias": "conv2_bias",
        "conv3.weight": "conv3_weight",
        "conv3.bias": "conv3_bias",
    }
    return aliases.get(name, name.replace(".", "_"))


def load_cifar10_batch(batch_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取 CIFAR-10 的一个批次，图像保持为原始 uint8 格式。"""
    with batch_path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")

    data_key = b"data" if b"data" in batch else "data"
    labels_key = b"labels" if b"labels" in batch else "labels"
    images = batch[data_key].reshape(-1, 3, 32, 32).astype(np.uint8)
    labels = np.asarray(batch[labels_key], dtype=np.int64)
    return images, labels


def load_cifar10_test(root: Path, max_samples: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """只读取测试集，便于做定点推理和退化曲线测试。"""
    images, labels = load_cifar10_batch(root / "test_batch")
    if max_samples is not None and max_samples > 0:
        images = images[:max_samples]
        labels = labels[:max_samples]
    return images, labels


def normalize_uint8_images(images: np.ndarray) -> np.ndarray:
    """把原始 CIFAR-10 图像转成与训练阶段一致的标准化浮点输入。"""
    images = images.astype(np.float32) / 255.0
    return (images - CIFAR10_MEAN[None, :, None, None]) / CIFAR10_STD[None, :, None, None]


def export_quantized_parameters(state_dict, save_path: Path, frac_bits: int = 16) -> None:
    """把 PyTorch 的浮点参数导出为 32bit 定点参数文件。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_params: Dict[str, np.ndarray] = {
        "frac_bits": np.array(frac_bits, dtype=np.int32),
        "scale": np.array(1 << frac_bits, dtype=np.int32),
        "mean": CIFAR10_MEAN.astype(np.float32),
        "std": CIFAR10_STD.astype(np.float32),
    }

    for name, value in state_dict.items():
        fixed_params[_canonical_param_name(name)] = quantize_to_fixed_point(value, frac_bits)

    np.savez_compressed(save_path, **fixed_params)


def _load_state_dict_to_quantized_arrays(state_dict, frac_bits: int) -> Dict[str, np.ndarray]:
    """把浮点 state_dict 转成适合手工定点仿真的整数参数。"""
    quantized: Dict[str, np.ndarray] = {}
    for name, value in state_dict.items():
        quantized[_canonical_param_name(name)] = quantize_to_fixed_point(value, frac_bits)
    return quantized


@dataclass
class FixedPointCIFAR10CNN:
    """纯 NumPy 的定点卷积网络推理器，不依赖 PyTorch 进行前向传播。"""

    conv1_weight: np.ndarray
    conv1_bias: np.ndarray
    conv2_weight: np.ndarray
    conv2_bias: np.ndarray
    conv3_weight: np.ndarray
    conv3_bias: np.ndarray
    classifier_weight: np.ndarray
    classifier_bias: np.ndarray
    frac_bits: int = 16
    input_frac_bits: int = 16

    @classmethod
    def from_state_dict(cls, state_dict, frac_bits: int = 16, input_frac_bits: int | None = None):
        """把训练好的 PyTorch 参数直接转成定点推理器。"""
        if input_frac_bits is None:
            input_frac_bits = frac_bits

        quantized = _load_state_dict_to_quantized_arrays(state_dict, frac_bits)
        return cls(
            conv1_weight=quantized["conv1_weight"],
            conv1_bias=quantized["conv1_bias"],
            conv2_weight=quantized["conv2_weight"],
            conv2_bias=quantized["conv2_bias"],
            conv3_weight=quantized["conv3_weight"],
            conv3_bias=quantized["conv3_bias"],
            classifier_weight=quantized["classifier_weight"],
            classifier_bias=quantized["classifier_bias"],
            frac_bits=frac_bits,
            input_frac_bits=input_frac_bits,
        )

    @classmethod
    def from_npz(cls, npz_path: Path):
        """直接从训练脚本导出的 32bit 定点参数文件构建模型。"""
        data = np.load(npz_path)
        frac_bits = int(data["frac_bits"])
        input_frac_bits = int(data["frac_bits"]) if "input_frac_bits" not in data else int(data["input_frac_bits"])

        def _pick(*names: str) -> np.ndarray:
            for name in names:
                if name in data:
                    return data[name]
            raise KeyError(f"定点参数文件缺少字段: {names[0]}")

        return cls(
            conv1_weight=_pick("conv1_weight", "features_0_weight"),
            conv1_bias=_pick("conv1_bias", "features_0_bias"),
            conv2_weight=_pick("conv2_weight", "features_3_weight"),
            conv2_bias=_pick("conv2_bias", "features_3_bias"),
            conv3_weight=_pick("conv3_weight", "features_6_weight"),
            conv3_bias=_pick("conv3_bias", "features_6_bias"),
            classifier_weight=_pick("classifier_weight", "classifier_weight"),
            classifier_bias=_pick("classifier_bias", "classifier_bias"),
            frac_bits=frac_bits,
            input_frac_bits=input_frac_bits,
        )

    def _quantize_activation(self, activation: np.ndarray) -> np.ndarray:
        """把浮点输入变成统一 Q 格式的整数激活值。"""
        scale = 1 << self.input_frac_bits
        quantized = np.round(activation.astype(np.float64) * scale)
        return _clip_to_int32(quantized)

    def _relu(self, values: np.ndarray) -> np.ndarray:
        """ReLU 在整数域里就是把所有负数截断到 0。"""
        return np.maximum(values, 0, out=values)

    def _max_pool2x2(self, values: np.ndarray) -> np.ndarray:
        """2x2 最大池化。由于 CIFAR-10 尺寸固定，这里写得非常直接。"""
        batch, channels, height, width = values.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("2x2 池化要求特征图高宽必须是偶数。")
        return values.reshape(batch, channels, height // 2, 2, width // 2, 2).max(axis=(3, 5))

    def _global_avg_pool(self, values: np.ndarray) -> np.ndarray:
        """全局平均池化，手工做整数求和再除法，保持定点风格。"""
        batch, channels, height, width = values.shape
        summed = values.astype(np.int64).sum(axis=(2, 3))
        averaged = summed // (height * width)
        return _clip_to_int32(averaged).reshape(batch, channels, 1, 1)

    def _conv2d(self, inputs: np.ndarray, weight: np.ndarray, bias: np.ndarray, padding: int = 1) -> np.ndarray:
        """纯 NumPy 卷积实现，借助 sliding_window_view 减少 Python 层循环。"""
        batch, in_channels, height, width = inputs.shape
        out_channels, weight_in_channels, kernel_h, kernel_w = weight.shape

        if in_channels != weight_in_channels:
            raise ValueError("输入通道数与卷积核通道数不匹配。")

        padded = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
        windows = sliding_window_view(padded, (kernel_h, kernel_w), axis=(2, 3))

        # windows 的形状是 [N, C_in, H_out, W_out, K_h, K_w]，直接做张量点积即可。
        conv = np.tensordot(
            windows.astype(np.int64),
            weight.astype(np.int64),
            axes=([1, 4, 5], [1, 2, 3]),
        )
        conv = np.moveaxis(conv, -1, 1)

        # 卷积输入和卷积核都属于 Q 格式，因此乘积后的小数位会翻倍，必须右移回原尺度。
        conv = (conv >> self.frac_bits) + bias[None, :, None, None].astype(np.int64)
        return _clip_to_int32(conv)

    def _linear(self, inputs: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """全连接层的整数实现。"""
        outputs = inputs.astype(np.int64) @ weight.astype(np.int64).T
        outputs = (outputs >> self.frac_bits) + bias[None, :].astype(np.int64)
        return _clip_to_int32(outputs)

    def forward_batch(self, images: np.ndarray) -> np.ndarray:
        """输入一批原始 CIFAR-10 图像，输出整数 logits。"""
        if images.ndim == 3:
            images = images[None, ...]

        # 先按训练阶段同样的方式做标准化，再量化为整数。
        normalized = normalize_uint8_images(images)
        activations = self._quantize_activation(normalized)

        activations = self._conv2d(activations, self.conv1_weight, self.conv1_bias, padding=1)
        activations = self._relu(activations)
        activations = self._max_pool2x2(activations)

        activations = self._conv2d(activations, self.conv2_weight, self.conv2_bias, padding=1)
        activations = self._relu(activations)
        activations = self._max_pool2x2(activations)

        activations = self._conv2d(activations, self.conv3_weight, self.conv3_bias, padding=1)
        activations = self._relu(activations)
        activations = self._global_avg_pool(activations)

        activations = activations.reshape(activations.shape[0], -1)
        logits = self._linear(activations, self.classifier_weight, self.classifier_bias)
        return logits

    def predict(self, images: np.ndarray) -> np.ndarray:
        """返回预测类别编号。"""
        return np.argmax(self.forward_batch(images), axis=1)


def evaluate_accuracy(model: FixedPointCIFAR10CNN, images: np.ndarray, labels: np.ndarray, batch_size: int = 128) -> float:
    """在测试集上统计定点推理准确率。"""
    total = 0
    correct = 0

    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        predictions = model.predict(images[start:end])
        correct += int((predictions == labels[start:end]).sum())
        total += end - start

    return correct / total