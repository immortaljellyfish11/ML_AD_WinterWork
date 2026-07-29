from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QuantSpec:
    weight_bits: int = 8
    activation_bits: int = 8
    accumulator_bits: int = 32
    frac_bits: int = 7

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits


def quant_spec_from_config(config: dict[str, Any]) -> QuantSpec:
    values = config["quantization"]
    return QuantSpec(
        weight_bits=int(values["weight_bits"]),
        activation_bits=int(values["activation_bits"]),
        accumulator_bits=int(values["accumulator_bits"]),
        frac_bits=int(values["frac_bits"]),
    )


def quantize_tensor(values: torch.Tensor, frac_bits: int, bits: int) -> torch.Tensor:
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    return torch.clamp(torch.round(values * float(1 << frac_bits)), lower, upper)


def quantize_input(values: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    return quantize_tensor(values, spec.frac_bits, spec.activation_bits)


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d | None) -> tuple[torch.Tensor, torch.Tensor]:
    weight = conv.weight
    if conv.bias is None:
        bias = torch.zeros(conv.out_channels, device=weight.device, dtype=weight.dtype)
    else:
        bias = conv.bias
    if bn is None:
        return weight, bias
    inv_std = torch.rsqrt(bn.running_var + bn.eps)
    gain = bn.weight * inv_std
    folded_weight = weight * gain.reshape(-1, 1, 1, 1)
    folded_bias = (bias - bn.running_mean) * gain + bn.bias
    return folded_weight, folded_bias


def _requantize(accumulator: torch.Tensor, spec: QuantSpec, relu: bool) -> torch.Tensor:
    values = torch.floor(accumulator / float(spec.scale))
    if relu:
        return torch.clamp(values, 0, (1 << (spec.activation_bits - 1)) - 1)
    return torch.clamp(
        values,
        -(1 << (spec.activation_bits - 1)),
        (1 << (spec.activation_bits - 1)) - 1,
    )


def qconv2d(
    inputs: torch.Tensor,
    conv: nn.Conv2d,
    spec: QuantSpec,
    *,
    bn: nn.BatchNorm2d | None = None,
    relu: bool = True,
) -> torch.Tensor:
    weight, bias = fold_conv_bn(conv, bn)
    weight_q = quantize_tensor(weight, spec.frac_bits, spec.weight_bits)
    bias_q = quantize_tensor(bias, spec.frac_bits * 2, spec.accumulator_bits)
    accumulator = F.conv2d(
        inputs,
        weight_q,
        bias_q,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )
    return _requantize(accumulator, spec, relu)


def qlinear(inputs: torch.Tensor, linear: nn.Linear, spec: QuantSpec) -> torch.Tensor:
    weight_q = quantize_tensor(linear.weight, spec.frac_bits, spec.weight_bits)
    accumulator = F.linear(inputs, weight_q, None)
    shifted = torch.floor(accumulator / float(spec.scale))
    if linear.bias is not None:
        shifted = shifted + quantize_tensor(linear.bias, spec.frac_bits, spec.accumulator_bits)
    return shifted


def qmaxpool2d(inputs: torch.Tensor, kernel_size: int = 2, stride: int = 2) -> torch.Tensor:
    return F.max_pool2d(inputs, kernel_size=kernel_size, stride=stride)


def qglobal_avg_pool(inputs: torch.Tensor) -> torch.Tensor:
    summed = inputs.sum(dim=(-2, -1), keepdim=True)
    divisor = inputs.shape[-2] * inputs.shape[-1]
    return torch.floor(summed / float(divisor))


def qresidual_relu(main: torch.Tensor, skip: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    return torch.clamp(main + skip, 0, (1 << (spec.activation_bits - 1)) - 1)


def _signed_hex(value: int, bits: int) -> str:
    width = (bits + 3) // 4
    return f"{value & ((1 << bits) - 1):0{width}x}"


def _write_mem(path: Path, values: torch.Tensor, bits: int) -> None:
    array = values.detach().cpu().to(torch.int64).numpy().reshape(-1)
    path.write_text("\n".join(_signed_hex(int(value), bits) for value in array) + "\n", encoding="ascii")


def export_int8_model(model: nn.Module, output_dir: Path, spec: QuantSpec, topology: list[dict[str, Any]]) -> Path:
    if not hasattr(model, "rtl_export_layers"):
        raise TypeError("model must implement rtl_export_layers()")
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_manifest: list[dict[str, Any]] = []
    for item in model.rtl_export_layers():
        name = str(item["name"])
        kind = str(item["kind"])
        module = item["module"]
        bn = item.get("bn")
        if kind in {"conv2d", "depthwise_conv2d", "pointwise_conv2d", "shortcut_conv2d"}:
            weight, bias = fold_conv_bn(module, bn)
            weight_q = quantize_tensor(weight, spec.frac_bits, spec.weight_bits)
            bias_q = quantize_tensor(bias, spec.frac_bits * 2, spec.accumulator_bits)
        elif kind == "linear":
            weight_q = quantize_tensor(module.weight, spec.frac_bits, spec.weight_bits)
            bias_value = module.bias if module.bias is not None else torch.zeros(module.out_features, device=weight_q.device)
            bias_q = quantize_tensor(bias_value, spec.frac_bits, spec.accumulator_bits)
        else:
            raise ValueError(f"unsupported export layer kind: {kind}")
        weight_file = f"{name}_weight.mem"
        bias_file = f"{name}_bias.mem"
        _write_mem(output_dir / weight_file, weight_q, spec.weight_bits)
        _write_mem(output_dir / bias_file, bias_q, spec.accumulator_bits)
        layer_manifest.append(
            {
                "name": name,
                "kind": kind,
                "weight_file": weight_file,
                "bias_file": bias_file,
                "weight_shape": list(weight_q.shape),
                "bias_shape": list(bias_q.shape),
                "stride": list(module.stride) if isinstance(module, nn.Conv2d) else None,
                "padding": list(module.padding) if isinstance(module, nn.Conv2d) else None,
                "groups": int(module.groups) if isinstance(module, nn.Conv2d) else None,
            }
        )
    manifest = {
        "format": "readmemh two's-complement",
        "quantization": spec.__dict__,
        "bias_scale": {
            "convolution": f"2^(2*{spec.frac_bits})",
            "linear": f"2^{spec.frac_bits}",
        },
        "topology": topology,
        "parameter_layers": layer_manifest,
    }
    (output_dir / "network_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir

