from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


RTL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RTL_ROOT.parent
DATA_DIR = RTL_ROOT / "sim" / "data"

# 必须与训练脚本 main.py / fixed_point_simulator.py 中的归一化参数一致。
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float64)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 edge_cnn_top 完整 CNN 顶层仿真输入和 golden")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "cifar-10-batches-py",
        help="CIFAR-10 原始数据目录",
    )
    parser.add_argument(
        "--param-dir",
        type=Path,
        default=DATA_DIR / "fpga_params" / "q07",
        help="由 accuracy_degradation_curve.py 导出的 FPGA 参数目录",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="从 CIFAR-10 test_batch 选取的起始样本编号")
    parser.add_argument("--num-samples", type=int, default=1, help="生成多少张测试图片")
    parser.add_argument("--frac-bits", type=int, default=7, help="必须与 edge_cnn_top 的 FRAC_BITS 一致")
    return parser.parse_args()


def load_cifar10_test(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 CIFAR-10 test_batch，返回 CHW 排列的 uint8 图片和标签。"""
    with (data_root / "test_batch").open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")

    data_key = b"data" if b"data" in batch else "data"
    labels_key = b"labels" if b"labels" in batch else "labels"
    images = batch[data_key].reshape(-1, 3, 32, 32).astype(np.uint8)
    labels = np.asarray(batch[labels_key], dtype=np.int64)
    return images, labels


def twos_complement_to_int(token: str, width: int) -> int:
    """把 .mem 中的十六进制二进制补码还原成 Python 有符号整数。"""
    raw = int(token, 16)
    sign_bit = 1 << (width - 1)
    full_range = 1 << width
    return raw - full_range if raw & sign_bit else raw


def read_mem(path: Path, width: int, shape: tuple[int, ...]) -> np.ndarray:
    """读取 readmemh 格式的一行一个参数文件，并 reshape 成神经网络需要的形状。"""
    values = [twos_complement_to_int(token, width) for token in path.read_text(encoding="ascii").split()]
    expected = int(np.prod(shape))
    if len(values) != expected:
        raise ValueError(f"{path} 参数数量不匹配: got {len(values)}, expected {expected}")
    return np.asarray(values, dtype=np.int64).reshape(shape)


def saturate_int8(value: int) -> int:
    """模拟 RTL saturate_int8.v，把结果夹到 signed INT8 范围。"""
    return max(-128, min(127, int(value)))


def quantize_image_to_int8(image: np.ndarray, frac_bits: int) -> np.ndarray:
    """把一张 CIFAR-10 uint8 图片变成 RTL 输入 buffer 需要的 signed INT8。

    训练时图片先除以 255，再做 mean/std 标准化。
    RTL 顶层输入口只有 8 bit，所以这里会做 INT8 饱和。
    """
    normalized = image.astype(np.float64) / 255.0
    normalized = (normalized - CIFAR10_MEAN[:, None, None]) / CIFAR10_STD[:, None, None]
    quantized = np.rint(normalized * (1 << frac_bits)).astype(np.int64)
    return np.clip(quantized, -128, 127).astype(np.int64)


def conv2d_relu_int8(inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray, frac_bits: int) -> np.ndarray:
    """模拟 conv_layer_engine_sync: 3x3 padding=1 卷积、加 bias、右移、ReLU、INT8 饱和。

    inputs:  [IC, H, W], signed INT8
    weights: [OC, IC, 3, 3], signed INT8，顺序与 RTL conv3x3_addr_gen 一致
    biases:  [OC], signed INT32。注意这里的卷积 bias 已按 2*frac_bits 缩放。
    """
    in_channels, height, width = inputs.shape
    out_channels = weights.shape[0]
    outputs = np.zeros((out_channels, height, width), dtype=np.int64)

    for oc in range(out_channels):
        for row in range(height):
            for col in range(width):
                total = int(biases[oc])
                for ic in range(in_channels):
                    for kr in range(3):
                        for kc in range(3):
                            src_row = row + kr - 1
                            src_col = col + kc - 1
                            if 0 <= src_row < height and 0 <= src_col < width:
                                total += int(inputs[ic, src_row, src_col]) * int(weights[oc, ic, kr, kc])

                quantized = total >> frac_bits
                relu = max(0, quantized)
                outputs[oc, row, col] = saturate_int8(relu)

    return outputs


def maxpool2x2(inputs: np.ndarray) -> np.ndarray:
    """模拟 pool_layer_2x2_engine_sync 的 2x2 stride=2 maxpool。"""
    channels, height, width = inputs.shape
    outputs = np.zeros((channels, height // 2, width // 2), dtype=np.int64)
    for ch in range(channels):
        for row in range(height // 2):
            for col in range(width // 2):
                window = inputs[ch, row * 2 : row * 2 + 2, col * 2 : col * 2 + 2]
                outputs[ch, row, col] = int(window.max())
    return outputs


def global_avg_pool8x8(inputs: np.ndarray) -> np.ndarray:
    """模拟 global_avg_pool_layer_engine_sync: 每个通道 sum(64) >>> 6。"""
    channels, height, width = inputs.shape
    if height != 8 or width != 8:
        raise ValueError("当前 RTL 顶层的 GAP 只对应 8x8 输入")
    outputs = np.zeros((channels,), dtype=np.int64)
    for ch in range(channels):
        outputs[ch] = saturate_int8(int(inputs[ch].sum()) >> 6)
    return outputs


def linear_128x10(features: np.ndarray, weights: np.ndarray, biases: np.ndarray, frac_bits: int) -> np.ndarray:
    """模拟 linear_128x10_sync: 128 项乘加，右移 frac_bits 后加 linear bias。"""
    logits = np.zeros((10,), dtype=np.int64)
    for cls in range(10):
        acc = 0
        for index in range(128):
            acc += int(features[index]) * int(weights[cls, index])
        logits[cls] = (acc >> frac_bits) + int(biases[cls])
    return logits


def load_params(param_dir: Path) -> dict[str, np.ndarray]:
    """读取当前 RTL ROM 会加载的同一批 .mem 参数。"""
    return {
        "conv1_weight": read_mem(param_dir / "conv1_weight.mem", 8, (32, 3, 3, 3)),
        "conv2_weight": read_mem(param_dir / "conv2_weight.mem", 8, (64, 32, 3, 3)),
        "conv3_weight": read_mem(param_dir / "conv3_weight.mem", 8, (128, 64, 3, 3)),
        "linear_weight": read_mem(param_dir / "linear_weight.mem", 8, (10, 128)),
        "conv1_bias": read_mem(param_dir / "conv1_bias.mem", 32, (32,)),
        "conv2_bias": read_mem(param_dir / "conv2_bias.mem", 32, (64,)),
        "conv3_bias": read_mem(param_dir / "conv3_bias.mem", 32, (128,)),
        "linear_bias": read_mem(param_dir / "linear_bias.mem", 32, (10,)),
    }


def forward_edge_cnn(image_int8: np.ndarray, params: dict[str, np.ndarray], frac_bits: int) -> np.ndarray:
    """用 Python 高层模型模拟当前 edge_cnn_top 的数据路径。"""
    activations = conv2d_relu_int8(image_int8, params["conv1_weight"], params["conv1_bias"], frac_bits)
    activations = maxpool2x2(activations)
    activations = conv2d_relu_int8(activations, params["conv2_weight"], params["conv2_bias"], frac_bits)
    activations = maxpool2x2(activations)
    activations = conv2d_relu_int8(activations, params["conv3_weight"], params["conv3_bias"], frac_bits)
    features = global_avg_pool8x8(activations)
    return linear_128x10(features, params["linear_weight"], params["linear_bias"], frac_bits)


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    images, labels = load_cifar10_test(args.data_root)
    params = load_params(args.param_dir)

    input_path = DATA_DIR / "edge_cnn_top_input.txt"
    golden_path = DATA_DIR / "edge_cnn_top_golden.txt"

    with input_path.open("w", encoding="utf-8") as input_file, golden_path.open("w", encoding="utf-8") as golden_file:
        for offset in range(args.num_samples):
            sample_index = args.sample_index + offset
            image_int8 = quantize_image_to_int8(images[sample_index], args.frac_bits)
            logits = forward_edge_cnn(image_int8, params, args.frac_bits)
            class_id = int(np.argmax(logits))
            max_logit = int(logits[class_id])
            label = int(labels[sample_index])

            input_values = image_int8.reshape(-1)
            input_file.write(" ".join(str(int(value)) for value in input_values) + "\n")
            golden_values = [label, class_id, max_logit, *[int(value) for value in logits]]
            golden_file.write(" ".join(str(value) for value in golden_values) + "\n")

            print(f"sample={sample_index} label={label} golden_class={class_id} max_logit={max_logit}")

    print(f"input:  {input_path}")
    print(f"golden: {golden_path}")


if __name__ == "__main__":
    main()
