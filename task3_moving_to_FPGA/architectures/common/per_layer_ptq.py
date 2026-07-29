"""Per-layer symmetric INT8 PTQ shared by the isolated CNN designs.

中文：本模块是 Python golden model 与 RTL 参数导出的唯一量化定义。
English: This module is the single quantization definition for Python and RTL.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_scale(max_abs: float) -> float:
    return max_abs / 127.0 if max_abs > 0.0 else 1.0


def quantize_symmetric(values: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.clamp(torch.round(values / scale), -127, 127)


def choose_multiplier_shift(real_multiplier: float) -> tuple[int, int]:
    if real_multiplier <= 0.0:
        raise ValueError("requantization multiplier must be positive")
    shift = 24
    multiplier = int(round(real_multiplier * (1 << shift)))
    # DSP48E1 implementation uses a signed 18-bit constant multiplier.
    # RTL 的 DSP48E1 常数乘数输入为有符号 18 位，因此主动降低 shift 直至可表示。
    while multiplier > (1 << 17) - 1:
        shift -= 1
        multiplier = int(round(real_multiplier * (1 << shift)))
    while multiplier == 0 and shift < 30:
        shift += 1
        multiplier = int(round(real_multiplier * (1 << shift)))
    if not 0 < multiplier <= (1 << 17) - 1:
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


def _module_map(model: nn.Module) -> dict[str, nn.Module]:
    return {item["name"]: item["module"] for item in model.rtl_export_layers()}


def _export_map(model: nn.Module) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in model.rtl_export_layers()}


def _source_name(item: dict[str, Any], previous: str) -> str:
    return str(item.get("input_from", previous))


def _fold_batch_norm(module: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold BN into Conv / 将 BatchNorm 折叠进卷积，得到推理时的 W' 和 b'。"""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = module.weight * scale.reshape(-1, 1, 1, 1)
    bias = bn.bias - scale * bn.running_mean
    if module.bias is not None:
        bias = bias + scale * module.bias
    return weight, bias


def _float_conv(outputs: torch.Tensor, item: dict[str, Any], exported: dict[str, dict[str, Any]]) -> torch.Tensor:
    entry = exported[item["name"]]
    module = entry["module"]
    bn = entry.get("bn")
    if bn is not None:
        weight, bias = _fold_batch_norm(module, bn)
    else:
        weight = module.weight
        bias = module.bias
    result = F.conv2d(outputs, weight, bias, stride=module.stride,
                      padding=module.padding, groups=module.groups)
    return torch.relu(result) if item.get("relu", True) else result


def _run_float_op(outputs: torch.Tensor, item: dict[str, Any], exported: dict[str, dict[str, Any]]) -> torch.Tensor:
    op = item["op"]
    if "conv" in op or "pointwise" in op or "depthwise" in op:
        return _float_conv(outputs, item, exported)
    if op == "maxpool2x2":
        return F.max_pool2d(outputs, 2, 2)
    if op == "residual_add_relu":
        return torch.relu(outputs + item["skip_tensor"])
    if op == "global_average_pool":
        return F.adaptive_avg_pool2d(outputs, (1, 1))
    if op == "linear":
        return exported[item["name"]]["module"](torch.flatten(outputs, 1))
    raise ValueError(f"unsupported PTQ op: {op}")


@torch.no_grad()
def calibrate(model: nn.Module, loader, topology: list[dict[str, Any]], device: torch.device,
              sample_limit: int = 2048, percentile: float = 99.9) -> dict[str, Any]:
    """Collect sampled absolute activation percentiles / 收集激活绝对值分位点。"""
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    exported = _export_map(model)
    conv_names = [item["name"] for item in topology if "conv" in item["op"] or "pointwise" in item["op"] or "depthwise" in item["op"]]
    activation_names = [item["name"] for item in topology if item["op"] in {
        "residual_add_relu", "global_average_pool", "maxpool2x2"
    }]
    names = ["input", *conv_names, *activation_names]
    maxima = {name: 0.0 for name in names}
    samples: dict[str, list[torch.Tensor]] = {name: [] for name in names}

    def observe(name: str, values: torch.Tensor) -> None:
        absolute = values.detach().abs().reshape(-1)
        maxima[name] = max(maxima[name], float(absolute.max().item()))
        stride = max(1, math.ceil(absolute.numel() / 32768))
        samples[name].append(absolute[::stride][:32768].cpu())

    processed = 0
    for images, _ in loader:
        if processed >= sample_limit:
            break
        keep = min(len(images), sample_limit - processed)
        current = images[:keep].to(device, non_blocking=True)
        tensors = {"input": current}
        observe("input", current)
        previous = "input"
        for item in topology:
            if item["op"] == "linear":
                break
            source = _source_name(item, previous)
            if item["op"] == "residual_add_relu":
                item = dict(item)
                item["skip_tensor"] = tensors[item["skip_from"]]
                result = _run_float_op(tensors[source], item, exported)
            else:
                result = _run_float_op(tensors[source], item, exported)
            tensors[item["name"]] = result
            observe(item["name"], result)
            previous = item["name"]
        processed += keep

    thresholds = {
        name: (maxima[name] if percentile == 100.0 else float(torch.quantile(torch.cat(samples[name]), percentile / 100.0)))
        for name in names
    }
    return {
        "samples": processed,
        "percentile": percentile,
        "maxima": maxima,
        "thresholds": thresholds,
        "scales": {name: symmetric_scale(value) for name, value in thresholds.items()},
    }


def build_parameters(model: nn.Module, topology: list[dict[str, Any]], calibration: dict[str, Any]):
    exported = _export_map(model)
    arrays: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, float | int]] = {}
    previous = "input"
    for item in topology:
        op = item["op"]
        if "conv" in op or "pointwise" in op or "depthwise" in op:
            name = item["name"]
            entry = exported[name]
            module = entry["module"]
            input_name = _source_name(item, previous)
            input_scale = float(calibration["scales"][input_name])
            output_scale = float(calibration["scales"][name])
            if entry.get("bn") is not None:
                weight, bias = _fold_batch_norm(module, entry["bn"])
            else:
                weight, bias = module.weight.detach(), module.bias.detach()
            weight, bias = weight.detach(), bias.detach()
            weight_scale = symmetric_scale(float(weight.abs().max()))
            bias_scale = input_scale * weight_scale
            multiplier, shift = choose_multiplier_shift(bias_scale / output_scale)
            arrays[f"{name}_weight"] = quantize_symmetric(weight, weight_scale).to(torch.int8)
            arrays[f"{name}_bias"] = torch.round(bias / bias_scale).clamp(-(1 << 31), (1 << 31) - 1).to(torch.int32)
            layers[name] = {
                "input_scale": input_scale, "weight_scale": weight_scale,
                "bias_scale": bias_scale, "output_scale": output_scale,
                "requant_multiplier": multiplier, "requant_shift": shift,
                "stride": int(module.stride[0]), "padding": int(module.padding[0]),
                "groups": int(module.groups), "relu": bool(item.get("relu", True)),
            }
            previous = name
        elif op == "residual_add_relu":
            name = item["name"]
            main_name = _source_name(item, previous)
            skip_name = str(item["skip_from"])
            output_scale = float(calibration["scales"][name])
            main_scale = float(calibration["scales"][main_name])
            skip_scale = float(calibration["scales"][skip_name])
            main_multiplier, main_shift = choose_multiplier_shift(main_scale / output_scale)
            skip_multiplier, skip_shift = choose_multiplier_shift(skip_scale / output_scale)
            layers[name] = {
                "main_scale": main_scale, "skip_scale": skip_scale, "output_scale": output_scale,
                "main_multiplier": main_multiplier, "main_shift": main_shift,
                "skip_multiplier": skip_multiplier, "skip_shift": skip_shift,
            }
            previous = name
        elif op in ("maxpool2x2", "global_average_pool"):
            previous = item["name"]
        elif item["op"] == "linear":
            name = item["name"]
            module = exported[name]["module"]
            input_name = _source_name(item, previous)
            input_scale = float(calibration["scales"][input_name])
            weight_scale = symmetric_scale(float(module.weight.detach().abs().max()))
            bias_scale = input_scale * weight_scale
            arrays[f"{name}_weight"] = quantize_symmetric(module.weight.detach(), weight_scale).to(torch.int8)
            arrays[f"{name}_bias"] = torch.round(module.bias.detach() / bias_scale).clamp(-(1 << 31), (1 << 31) - 1).to(torch.int32)
            layers[name] = {
                "input_scale": input_scale, "weight_scale": weight_scale,
                "bias_scale": bias_scale, "output_scale": bias_scale,
                "requant_multiplier": 1, "requant_shift": 0,
            }
            previous = name
    return arrays, layers


class IntegerCNN:
    """Topology-driven integer reference model / 由拓扑驱动的整数参考模型。"""
    def __init__(self, arrays: dict[str, torch.Tensor], quantization: dict[str, Any],
                 topology: list[dict[str, Any]], device: torch.device) -> None:
        self.arrays = {name: values.to(device) for name, values in arrays.items()}
        self.quantization = quantization
        self.topology = topology

    def quantize_input(self, images: torch.Tensor) -> torch.Tensor:
        return quantize_symmetric(images, float(self.quantization["activation_scales"]["input"])).to(torch.float32)

    def forward_quantized(self, outputs: torch.Tensor) -> torch.Tensor:
        tensors = {"input": outputs}
        previous = "input"
        for item in self.topology:
            name, op = item["name"], item["op"]
            if "conv" in op or "pointwise" in op or "depthwise" in op:
                source = _source_name(item, previous)
                params = self.quantization["layers"][name]
                accumulator = F.conv2d(tensors[source], self.arrays[f"{name}_weight"].float(),
                                       self.arrays[f"{name}_bias"].float(),
                                       stride=int(params["stride"]), padding=int(params["padding"]),
                                       groups=int(params["groups"]))
                outputs = requantize(accumulator, int(params["requant_multiplier"]),
                                     int(params["requant_shift"]))
                outputs = (outputs.clamp(0, 127) if bool(params["relu"]) else outputs.clamp(-127, 127)).float()
                tensors[name] = outputs
                previous = name
            elif op == "maxpool2x2":
                source = _source_name(item, previous)
                tensors[name] = F.max_pool2d(tensors[source], 2, 2)
                previous = name
            elif op == "residual_add_relu":
                source = _source_name(item, previous)
                params = self.quantization["layers"][name]
                main = requantize(tensors[source], int(params["main_multiplier"]), int(params["main_shift"]))
                skip = requantize(tensors[item["skip_from"]], int(params["skip_multiplier"]), int(params["skip_shift"]))
                tensors[name] = (main + skip).clamp(0, 127).float()
                previous = name
            elif op == "global_average_pool":
                area = int(item["input"][1]) * int(item["input"][2])
                source = _source_name(item, previous)
                tensors[name] = torch.floor(tensors[source].sum(dim=(-2, -1), keepdim=True) / area)
                previous = name
            elif op == "linear":
                source = _source_name(item, previous)
                outputs = F.linear(torch.flatten(tensors[source], 1), self.arrays[f"{name}_weight"].float(),
                                   self.arrays[f"{name}_bias"].float())
            else:
                raise ValueError(f"unsupported integer op: {op}")
        return outputs

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_quantized(self.quantize_input(images))


def signed_hex(value: int, bits: int) -> str:
    return f"{value & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def export(output_dir: Path, arrays: dict[str, torch.Tensor], quantization: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        bits = 8 if name.endswith("weight") else 32
        flat = values.detach().cpu().to(torch.int64).reshape(-1)
        (output_dir / f"{name}.mem").write_text(
            "\n".join(signed_hex(int(value), bits) for value in flat) + "\n", encoding="ascii")
    (output_dir / "quantization.json").write_text(json.dumps(quantization, indent=2), encoding="utf-8")
    manifest = [
        "scheme=per-layer symmetric INT8 PTQ", "weight_width=8", "activation_width=8",
        "bias_width=32", "accumulator_width=32", "rounding=nearest; ties away from zero",
        "relu_saturation=[0,127]",
    ]
    for name, params in quantization["layers"].items():
        if "requant_multiplier" in params:
            manifest.extend([f"{name}_requant_multiplier={params['requant_multiplier']}",
                             f"{name}_requant_shift={params['requant_shift']}"])
        elif "main_multiplier" in params:
            manifest.extend([
                f"{name}_main_multiplier={params['main_multiplier']}",
                f"{name}_main_shift={params['main_shift']}",
                f"{name}_skip_multiplier={params['skip_multiplier']}",
                f"{name}_skip_shift={params['skip_shift']}",
            ])
    (output_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    # Verilog include keeps XSim and synthesis parameters tied to this export.
    # Verilog 头文件保证 XSim/综合始终使用本次导出的重量化参数。
    header = ["`ifndef GENERATED_RTL_PARAMS_VH", "`define GENERATED_RTL_PARAMS_VH"]
    for name, params in quantization["layers"].items():
        if name.startswith("conv"):
            macro = name.upper()
            header.extend([f"`define Q_{macro}_M {params['requant_multiplier']}",
                           f"`define Q_{macro}_S {params['requant_shift']}"])
    header.append("`endif")
    (output_dir / "rtl_params.vh").write_text("\n".join(header) + "\n", encoding="ascii")
    tcl = ["# Generated by per_layer_ptq.py; do not edit manually."]
    for name, params in quantization["layers"].items():
        if name.startswith("conv"):
            macro = name.upper()
            tcl.extend([f"set Q_{macro}_M {params['requant_multiplier']}",
                        f"set Q_{macro}_S {params['requant_shift']}"])
    (output_dir / "rtl_params.tcl").write_text("\n".join(tcl) + "\n", encoding="ascii")
