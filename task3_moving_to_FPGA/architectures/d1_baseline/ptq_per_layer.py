from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architectures.common.data import CIFAR10Dataset, load_batch, load_splits, set_seed
from architectures.d1_baseline.model import BaselineCNN


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "cifar10_cnn_best.pth"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "cifar-10-batches-py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "rtl" / "sim" / "data" / "fpga_params" / "per_layer_int8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and export D1 per-layer symmetric INT8 PTQ")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-samples", type=int, default=2048)
    parser.add_argument("--activation-percentile", type=float, default=99.9)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(path: Path, device: torch.device) -> BaselineCNN:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model = BaselineCNN().to(device)
    model.load_state_dict(state)
    return model.eval()


def symmetric_scale(max_abs: float, quant_max: int = 127) -> float:
    if max_abs <= 0.0:
        return 1.0
    return max_abs / quant_max


def quantize_symmetric(values: torch.Tensor, scale: float, bits: int) -> torch.Tensor:
    quant_max = (1 << (bits - 1)) - 1
    quant_min = -quant_max
    return torch.clamp(torch.round(values / scale), quant_min, quant_max)


def quantize_bias(values: torch.Tensor, scale: float) -> torch.Tensor:
    info = torch.iinfo(torch.int32)
    return torch.clamp(torch.round(values / scale), info.min, info.max)


def choose_multiplier_shift(real_multiplier: float) -> tuple[int, int]:
    if real_multiplier <= 0.0:
        raise ValueError("requantization multiplier must be positive")
    # Q24 preserves the calibrated ratio while keeping the coefficient within
    # the DSP48E1 18-bit input once the INT32 accumulator is narrowed to its
    # proven 25-bit convolution range.
    # keeps the constant multiplier within one DSP48's 18-bit input.
    shift = 24
    multiplier = int(round(real_multiplier * (1 << shift)))
    while multiplier > (1 << 31) - 1:
        shift -= 1
        if shift < 0:
            raise OverflowError("requantization multiplier does not fit signed INT32")
        multiplier = int(round(real_multiplier * (1 << shift)))
    while multiplier == 0 and shift < 30:
        shift += 1
        multiplier = int(round(real_multiplier * (1 << shift)))
    if not 0 < multiplier <= (1 << 31) - 1:
        raise OverflowError("invalid requantization multiplier")
    return multiplier, shift


def requantize(values: torch.Tensor, multiplier: int, shift: int) -> torch.Tensor:
    product = values.to(torch.int64) * multiplier
    if shift == 0:
        return product
    offset = 1 << (shift - 1)
    positive = (product + offset) >> shift
    negative = -(((-product) + offset) >> shift)
    return torch.where(product >= 0, positive, negative)


@torch.no_grad()
def collect_activation_maxima(
    model: BaselineCNN,
    loader: DataLoader,
    device: torch.device,
    calibration_samples: int,
    percentile: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if not 0.0 < percentile <= 100.0:
        raise ValueError("activation percentile must be in (0, 100]")
    maxima = {"input": 0.0, "conv1": 0.0, "conv2": 0.0, "conv3": 0.0}
    samples: dict[str, list[torch.Tensor]] = {name: [] for name in maxima}

    def observe(name: str, values: torch.Tensor) -> None:
        absolute = values.detach().abs().reshape(-1)
        maxima[name] = max(maxima[name], float(absolute.max().item()))
        if percentile < 100.0:
            limit = 32768
            step = max(1, math.ceil(absolute.numel() / limit))
            samples[name].append(absolute[::step][:limit].cpu())

    processed = 0
    for images, _ in loader:
        if processed >= calibration_samples:
            break
        keep = min(len(images), calibration_samples - processed)
        images = images[:keep].to(device, non_blocking=True)
        observe("input", images)

        outputs = torch.relu(model.features[0](images))
        observe("conv1", outputs)
        outputs = F.max_pool2d(outputs, 2, 2)

        outputs = torch.relu(model.features[3](outputs))
        observe("conv2", outputs)
        outputs = F.max_pool2d(outputs, 2, 2)

        outputs = torch.relu(model.features[6](outputs))
        observe("conv3", outputs)
        processed += keep
    thresholds = dict(maxima)
    if percentile < 100.0:
        thresholds = {
            name: float(torch.quantile(torch.cat(values), percentile / 100.0).item())
            for name, values in samples.items()
        }
    maxima["samples"] = processed
    thresholds["samples"] = processed
    return thresholds, maxima


def build_quantized_parameters(model: BaselineCNN, activation_scales: dict[str, float]):
    layer_modules: list[tuple[str, nn.Module, str, str]] = [
        ("conv1", model.features[0], "input", "conv1"),
        ("conv2", model.features[3], "conv1", "conv2"),
        ("conv3", model.features[6], "conv2", "conv3"),
    ]
    arrays: dict[str, torch.Tensor] = {}
    layer_config: dict[str, dict[str, float | int]] = {}
    for name, module, input_name, output_name in layer_modules:
        weight_scale = symmetric_scale(float(module.weight.detach().abs().max().item()))
        input_scale = activation_scales[input_name]
        output_scale = activation_scales[output_name]
        bias_scale = input_scale * weight_scale
        multiplier, shift = choose_multiplier_shift(bias_scale / output_scale)
        arrays[f"{name}_weight"] = quantize_symmetric(module.weight.detach(), weight_scale, 8).to(torch.int8)
        arrays[f"{name}_bias"] = quantize_bias(module.bias.detach(), bias_scale).to(torch.int32)
        layer_config[name] = {
            "input_scale": input_scale,
            "weight_scale": weight_scale,
            "bias_scale": bias_scale,
            "output_scale": output_scale,
            "requant_multiplier": multiplier,
            "requant_shift": shift,
        }

    linear = model.classifier
    linear_weight_scale = symmetric_scale(float(linear.weight.detach().abs().max().item()))
    linear_input_scale = activation_scales["conv3"]
    linear_bias_scale = linear_input_scale * linear_weight_scale
    arrays["linear_weight"] = quantize_symmetric(linear.weight.detach(), linear_weight_scale, 8).to(torch.int8)
    arrays["linear_bias"] = quantize_bias(linear.bias.detach(), linear_bias_scale).to(torch.int32)
    layer_config["linear"] = {
        "input_scale": linear_input_scale,
        "weight_scale": linear_weight_scale,
        "bias_scale": linear_bias_scale,
        "output_scale": linear_bias_scale,
        "requant_multiplier": 1,
        "requant_shift": 0,
    }
    return arrays, layer_config


class IntegerD1:
    def __init__(self, arrays: dict[str, torch.Tensor], config: dict, device: torch.device) -> None:
        self.device = device
        self.config = config
        self.arrays = {name: value.to(device) for name, value in arrays.items()}

    def quantize_input(self, normalized_images: torch.Tensor) -> torch.Tensor:
        scale = float(self.config["activation_scales"]["input"])
        return quantize_symmetric(normalized_images, scale, 8).to(torch.float32)

    def conv(self, inputs: torch.Tensor, name: str, padding: int = 1) -> torch.Tensor:
        weight = self.arrays[f"{name}_weight"].to(torch.float32)
        bias = self.arrays[f"{name}_bias"].to(torch.float32)
        accumulator = F.conv2d(inputs, weight, bias, padding=padding)
        params = self.config["layers"][name]
        outputs = requantize(
            accumulator,
            int(params["requant_multiplier"]),
            int(params["requant_shift"]),
        )
        return torch.clamp(outputs, 0, 127).to(torch.float32)

    def forward_quantized(self, inputs_q: torch.Tensor) -> torch.Tensor:
        outputs = F.max_pool2d(self.conv(inputs_q, "conv1"), 2, 2)
        outputs = F.max_pool2d(self.conv(outputs, "conv2"), 2, 2)
        outputs = self.conv(outputs, "conv3")
        outputs = torch.floor(outputs.sum(dim=(-2, -1)) / 64.0)
        weight = self.arrays["linear_weight"].to(torch.float32)
        bias = self.arrays["linear_bias"].to(torch.float32)
        return F.linear(outputs, weight, bias)

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        return self.forward_quantized(self.quantize_input(normalized_images))


def signed_hex(value: int, bits: int) -> str:
    digits = (bits + 3) // 4
    return f"{value & ((1 << bits) - 1):0{digits}x}"


def write_mem(path: Path, values: torch.Tensor, bits: int) -> None:
    flat = values.detach().cpu().to(torch.int64).numpy().reshape(-1)
    path.write_text("\n".join(signed_hex(int(value), bits) for value in flat) + "\n", encoding="ascii")


def export(output_dir: Path, arrays: dict[str, torch.Tensor], config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        bits = 8 if name.endswith("weight") else 32
        write_mem(output_dir / f"{name}.mem", values, bits)
    (output_dir / "quantization.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    lines = [
        "scheme=per-layer symmetric INT8 PTQ",
        "weight_width=8",
        "activation_width=8",
        "bias_width=32",
        "accumulator_width=32",
        "rounding=nearest, ties away from zero",
        "saturation=signed INT8; post-ReLU [0,127]",
    ]
    for name, params in config["layers"].items():
        lines.append(f"{name}_requant_multiplier={params['requant_multiplier']}")
        lines.append(f"{name}_requant_shift={params['requant_shift']}")
    (output_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate(model: IntegerD1, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.forward(images)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    return correct / total


def main() -> None:
    args = parse_args()
    set_seed(9)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.checkpoint, device)
    splits = load_splits(args.data_root, val_size=7500)
    calibration_dataset = CIFAR10Dataset(splits[2], splits[3], train=False)
    calibration_loader = DataLoader(calibration_dataset, batch_size=args.batch_size, shuffle=False)
    thresholds, maxima = collect_activation_maxima(
        model,
        calibration_loader,
        device,
        args.calibration_samples,
        args.activation_percentile,
    )
    activation_scales = {
        name: symmetric_scale(value)
        for name, value in thresholds.items()
        if name != "samples"
    }
    arrays, layers = build_quantized_parameters(model, activation_scales)
    config = {
        "scheme": "per-layer symmetric INT8 PTQ",
        "calibration": {
            "samples": int(maxima["samples"]),
            "source": "CIFAR-10 validation split",
            "method": "sampled absolute percentile",
            "activation_percentile": args.activation_percentile,
        },
        "activation_maxima": {name: value for name, value in maxima.items() if name != "samples"},
        "activation_thresholds": {name: value for name, value in thresholds.items() if name != "samples"},
        "activation_scales": activation_scales,
        "layers": layers,
    }
    export(args.output_dir, arrays, config)
    print(json.dumps(config, indent=2))
    print(f"exported={args.output_dir}")

    if not args.skip_evaluation:
        test_images, test_labels = load_batch(args.data_root / "test_batch")
        test_loader = DataLoader(CIFAR10Dataset(test_images, test_labels, train=False), batch_size=args.batch_size)
        integer_model = IntegerD1(arrays, config, device)
        accuracy = evaluate(integer_model, test_loader, device)
        metrics = {"int8_test_accuracy": accuracy, "test_samples": len(test_labels)}
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
