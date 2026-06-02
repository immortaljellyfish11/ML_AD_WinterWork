from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pickle import UnpicklingError

from fixed_point_simulator import FixedPointCIFAR10CNN, evaluate_accuracy, load_cifar10_test


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 CIFAR-10 模型在不同定点精度下的准确率退化曲线")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "cifar-10-batches-py",
        help="CIFAR-10 原始数据目录",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "cifar10_cnn_best.pth",
        help="训练脚本保存的浮点模型权重",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="图像保存目录",
    )
    parser.add_argument("--bits-start", type=int, default=4, help="退化曲线起始小数位数")
    parser.add_argument("--bits-end", type=int, default=16, help="退化曲线结束小数位数")
    parser.add_argument("--bits-step", type=int, default=1, help="退化曲线步长")
    parser.add_argument("--batch-size", type=int, default=128, help="定点测试时的批大小")
    parser.add_argument("--max-samples", type=int, default=0, help="只取测试集前多少个样本，0 表示全部")
    parser.add_argument("--no-show", action="store_true", help="仅保存图片，不主动弹出窗口")
    parser.add_argument(
        "--no-export-fpga-params",
        action="store_true",
        help="只画精度曲线，不导出 Verilog ROM 可读的 FPGA 参数 .mem 文件",
    )
    parser.add_argument(
        "--fpga-param-dir",
        type=Path,
        default=None,
        help="FPGA 参数 .mem 文件输出目录；默认保存到 output-dir/fpga_params",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path):
    """兼容 PyTorch 2.6 之后的 weights_only 默认行为。"""
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (UnpicklingError, RuntimeError, AttributeError, ValueError):
        # 这个 checkpoint 是本地训练脚本生成的，内容可信时可以回退到完整反序列化。
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _state_array(state_dict, name: str) -> np.ndarray:
    """从 PyTorch state_dict 取出参数，并统一转成 NumPy 浮点数组。"""
    if name not in state_dict:
        raise KeyError(f"checkpoint 缺少参数: {name}")
    value = state_dict[name]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64)


def _quantize_signed(values: np.ndarray, frac_bits: int, width: int) -> np.ndarray:
    """按 Q 格式量化，并饱和到指定有符号位宽。"""
    scale = 1 << frac_bits
    quantized = np.rint(values * scale).astype(np.int64)
    min_value = -(1 << (width - 1))
    max_value = (1 << (width - 1)) - 1
    return np.clip(quantized, min_value, max_value).astype(np.int64)


def _signed_to_hex(value: int, width: int) -> str:
    """把有符号整数写成 readmemh 需要的二进制补码十六进制。"""
    mask = (1 << width) - 1
    digits = (width + 3) // 4
    return f"{int(value) & mask:0{digits}x}"


def _write_mem_file(path: Path, values: np.ndarray, width: int) -> None:
    """每行一个参数，便于 Verilog 的 $readmemh 直接加载。"""
    flat_values = np.asarray(values, dtype=np.int64).reshape(-1)
    path.write_text("\n".join(_signed_to_hex(value, width) for value in flat_values) + "\n", encoding="ascii")


def export_fpga_int8_params(state_dict, output_root: Path, frac_bits: int) -> Path:
    """导出当前 RTL 顶层可直接使用的 FPGA/Verilator 参数文件。

    当前 edge_cnn_top 里每一层都有独立的 weight_rom / bias_rom，
    因此这里按层分别保存 .mem 文件，而不是把所有参数塞进一个大文件。
    如果以后想换成单个大 ROM，也可以把这些文件合并后在 RTL 里增加 base address。

    权重格式:
        INT8 二进制补码，展平顺序与 RTL 地址生成一致。
        Conv:   OIHW, addr = oc * IC * 9 + ic * 9 + kr * 3 + kc
        Linear: out_class, in_feature

    bias 格式:
        INT32 二进制补码。
        Conv 当前 RTL 是 (raw_sum + bias) >>> FRAC_BITS，所以 Conv bias 使用 2*frac_bits 缩放。
        Linear 当前 RTL 是 (acc >>> FRAC_BITS) + bias，所以 Linear bias 使用 frac_bits 缩放。
    """
    export_dir = output_root / f"q{frac_bits:02d}"
    export_dir.mkdir(parents=True, exist_ok=True)

    weight_specs = [
        ("features.0.weight", "conv1_weight.mem"),
        ("features.3.weight", "conv2_weight.mem"),
        ("features.6.weight", "conv3_weight.mem"),
        ("classifier.weight", "linear_weight.mem"),
    ]
    for state_name, file_name in weight_specs:
        weights = _quantize_signed(_state_array(state_dict, state_name), frac_bits=frac_bits, width=8)
        _write_mem_file(export_dir / file_name, weights, width=8)

    conv_bias_specs = [
        ("features.0.bias", "conv1_bias.mem"),
        ("features.3.bias", "conv2_bias.mem"),
        ("features.6.bias", "conv3_bias.mem"),
    ]
    for state_name, file_name in conv_bias_specs:
        biases = _quantize_signed(_state_array(state_dict, state_name), frac_bits=frac_bits * 2, width=32)
        _write_mem_file(export_dir / file_name, biases, width=32)

    linear_bias = _quantize_signed(_state_array(state_dict, "classifier.bias"), frac_bits=frac_bits, width=32)
    _write_mem_file(export_dir / "linear_bias.mem", linear_bias, width=32)

    rtl_root = Path(__file__).resolve().parent / "rtl"
    try:
        verilator_dir = export_dir.resolve().relative_to(rtl_root.resolve()).as_posix()
    except ValueError:
        verilator_dir = export_dir.as_posix()

    manifest = "\n".join(
        [
            f"frac_bits={frac_bits}",
            "format=readmemh hex two's-complement",
            "weight_width=8",
            "bias_width=32",
            "rtl_note=current edge_cnn_top uses FRAC_BITS=7 by default; q07 matches current RTL directly",
            "conv_bias_scale=2^(2*frac_bits), because RTL does (raw_sum + bias) >>> FRAC_BITS",
            "linear_bias_scale=2^frac_bits, because RTL does (acc >>> FRAC_BITS) + bias",
            "",
            "Verilator -G 参数示例:",
            "# 下面路径默认从 task3_moving_to_FPGA/rtl 目录运行 verilator。",
            f"-GCONV1_WEIGHT_FILE='\"{verilator_dir}/conv1_weight.mem\"'",
            f"-GCONV2_WEIGHT_FILE='\"{verilator_dir}/conv2_weight.mem\"'",
            f"-GCONV3_WEIGHT_FILE='\"{verilator_dir}/conv3_weight.mem\"'",
            f"-GLINEAR_WEIGHT_FILE='\"{verilator_dir}/linear_weight.mem\"'",
            f"-GCONV1_BIAS_FILE='\"{verilator_dir}/conv1_bias.mem\"'",
            f"-GCONV2_BIAS_FILE='\"{verilator_dir}/conv2_bias.mem\"'",
            f"-GCONV3_BIAS_FILE='\"{verilator_dir}/conv3_bias.mem\"'",
            f"-GLINEAR_BIAS_FILE='\"{verilator_dir}/linear_bias.mem\"'",
            "",
        ]
    )
    (export_dir / "manifest.txt").write_text(manifest, encoding="utf-8")
    return export_dir


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"找不到模型权重文件: {args.checkpoint}")

    # 这里读取的是训练脚本保存的 PyTorch 浮点权重。后面会在不同的小数位数下反复量化。
    checkpoint = load_checkpoint(args.checkpoint)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    max_samples = args.max_samples if args.max_samples > 0 else None
    test_images, test_labels = load_cifar10_test(args.data_root, max_samples=max_samples)

    frac_bits_list = list(range(args.bits_start, args.bits_end + 1, args.bits_step))
    accuracies = []
    fpga_param_root = args.fpga_param_dir if args.fpga_param_dir is not None else args.output_dir / "fpga_params"

    print(f"测试样本数: {len(test_images)}")
    for frac_bits in frac_bits_list:
        # 每个精度都单独量化一遍参数，模拟 FPGA 里不同定点位宽带来的误差。
        model = FixedPointCIFAR10CNN.from_state_dict(state_dict, frac_bits=frac_bits)
        accuracy = evaluate_accuracy(model, test_images, test_labels, batch_size=args.batch_size)
        accuracies.append(accuracy)
        print(f"frac_bits={frac_bits:02d} -> accuracy={accuracy:.4f}")
        if not args.no_export_fpga_params:
            export_dir = export_fpga_int8_params(state_dict, fpga_param_root, frac_bits=frac_bits)
            print(f"  FPGA/Verilator 参数已导出: {export_dir}")
            if frac_bits != 7:
                print("  注意: 当前 edge_cnn_top 内部 FRAC_BITS 默认是 7；该参数集需同步修改 RTL 后才适合直接仿真。")


    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "accuracy_degradation_curve.png"

    plt.figure(figsize=(10, 6))
    plt.plot(frac_bits_list, accuracies, marker="o", linewidth=2)
    plt.xlabel("定点小数位数（frac bits）")
    plt.ylabel("测试集准确率")
    plt.title("CIFAR-10 定点精度退化曲线")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(frac_bits_list)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)

    if not args.no_show:
        plt.show()
    else:
        plt.close()

    print(f"退化曲线图片已保存到: {figure_path}")


if __name__ == "__main__":
    main()
