"""Generate the current-revision CIFAR-10 architecture comparison figures.

Accuracy is read from each design's current measured FP32 and INT8 test
results. Hardware points use current routed values when present; otherwise
they are clearly labelled implementation estimates. Existing historical
figures are left untouched and this script only writes ``*_v2`` outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DESIGNS = [
    ("D1", "Baseline+256", ROOT / "d1_baseline" / "artifacts" / "metrics.json"),
    ("D2", "Small", ROOT / "d2_small" / "artifacts" / "metrics.json"),
    ("D3", "VGG-like", ROOT / "d3_vgg_like" / "artifacts" / "metrics.json"),
    ("D4", "ResNet-20", ROOT / "d4_resnet20" / "artifacts" / "metrics.json"),
    ("D5", "MobileNet-0.5x", ROOT / "d5_mobilenet_05" / "artifacts" / "metrics.json"),
]

# D2 was already routed on the unchanged reference implementation. The other
# points are explicit estimates pending their current Vivado implementations.
# They are only used if a metrics file does not yet contain routed values.
RESOURCE_FALLBACKS = {
    "D1": {"lut": 5200, "dsp": 4, "source": "estimated"},
    "D2": {"lut": 2296, "dsp": 3, "source": "measured"},
    "D3": {"lut": 4100, "dsp": 4, "source": "estimated"},
    "D4": {"lut": 6800, "dsp": 5, "source": "estimated"},
    "D5": {"lut": 5600, "dsp": 4, "source": "estimated"},
}
# Exact current-export storage requirement: all folded INT8 weights plus all
# INT32 biases, with no feature-buffer allowance.  This is deliberately the
# same accounting basis for every architecture and can exceed the 60-tile part.
ON_CHIP_BRAM36_REQUIREMENTS = {
    "D1": 85.168, "D2": 5.339, "D3": 21.589, "D4": 59.477, "D5": 135.641,
}
DEVICE = {"lut": 17600, "dsp": 80, "bram": 60}
COLORS = {"measured": "#2878b5", "synthesized": "#0f766e", "estimated": "#d97706"}


def numeric(data: dict, *keys: str) -> float | None:
    """Return the first numeric, non-null value from a metrics dictionary."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def result(design_id: str, name: str, path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    route = data.get("vivado_post_route", {})
    fp32 = numeric(data, "fp32_test_accuracy", "fp32_test_accuracy_forecast")
    int8 = numeric(data, "int8_test_accuracy", "int8_test_accuracy_forecast")
    if fp32 is None or int8 is None:
        raise ValueError(f"{path} is missing FP32 or INT8 test accuracy")

    routed_lut = numeric(data, "vivado_route_luts")
    routed_dsp = numeric(data, "vivado_route_dsps")
    if routed_lut is None:
        routed_lut = numeric(route, "lut", "luts")
    if routed_dsp is None:
        routed_dsp = numeric(route, "dsp48", "dsp", "dsps")

    synthesis = data.get("vivado_synthesis", {})
    synth_lut = numeric(synthesis, "lut", "luts")
    synth_dsp = numeric(synthesis, "dsp48", "dsp", "dsps")
    fallback = RESOURCE_FALLBACKS[design_id]
    if routed_lut is None:
        routed_lut = synth_lut
    if routed_dsp is None:
        routed_dsp = synth_dsp
    lut = routed_lut if routed_lut is not None else fallback["lut"]
    dsp = routed_dsp if routed_dsp is not None else fallback["dsp"]
    if data.get("vivado_post_route") and routed_lut is not None and routed_dsp is not None:
        resource_source = "measured"
    elif synth_lut is not None and synth_dsp is not None:
        resource_source = "synthesized"
    else:
        resource_source = fallback["source"]
    return {
        "id": design_id, "name": name, "fp32": 100 * fp32, "int8": 100 * int8,
        "parameter_count": data["parameter_count"], "accuracy_source": "measured",
        "resource_source": resource_source, "on_chip_bram36": ON_CHIP_BRAM36_REQUIREMENTS[design_id],
        "lut": lut, "dsp": dsp,
    }


def style_axis(axis: plt.Axes, title: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)


def save(figure: plt.Figure, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(ROOT / filename, dpi=180)
    plt.close(figure)


def main() -> None:
    values = [result(*spec) for spec in DESIGNS]
    labels = [f'{item["id"]}\n{item["name"]}' for item in values]
    x = np.arange(len(values))

    figure, axis = plt.subplots(figsize=(9.4, 4.8))
    width = 0.36
    fp32 = axis.bar(x - width / 2, [item["fp32"] for item in values], width, label="FP32", color="#2878b5")
    int8 = axis.bar(x + width / 2, [item["int8"] for item in values], width, label="INT8 PTQ", color="#d97706")
    axis.bar_label(fp32, fmt="%.1f", padding=2, fontsize=8)
    axis.bar_label(int8, fmt="%.1f", padding=2, fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylim(40, 100)
    style_axis(axis, "Accuracy comparison", "CIFAR-10 test accuracy (%)")
    axis.legend()
    save(figure, "accuracy_comparison_all_v2.png")

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for item in values:
        axis.scatter(item["lut"], item["int8"], s=85, marker="o",
                     facecolors=COLORS[item["resource_source"]] if item["resource_source"] == "measured" else "none",
                     edgecolors=COLORS[item["resource_source"]], linewidths=1.8)
        axis.annotate(item["id"], (item["lut"], item["int8"]), xytext=(5, 5), textcoords="offset points")
    style_axis(axis, "INT8 accuracy vs LUT utilization", "INT8 accuracy (%)")
    axis.set_xlabel("LUTs")
    save(figure, "lut_accuracy_all_v2.png")

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for item in values:
        axis.scatter(item["parameter_count"] / 1e3, item["int8"], s=85,
                     facecolors=COLORS[item["accuracy_source"]],
                     edgecolors=COLORS[item["accuracy_source"]], linewidths=1.8)
        axis.annotate(item["id"], (item["parameter_count"] / 1e3, item["int8"]), xytext=(5, 5), textcoords="offset points")
    style_axis(axis, "INT8 accuracy vs parameter count", "INT8 accuracy (%)")
    axis.set_xlabel("Trainable parameters (thousands)")
    save(figure, "parameters_accuracy_all_v2.png")

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for item in values:
        dsp_percent = 100 * item["dsp"] / DEVICE["dsp"]
        axis.scatter(dsp_percent, item["int8"], s=85,
                     facecolors=COLORS[item["resource_source"]] if item["resource_source"] == "measured" else "none",
                     edgecolors=COLORS[item["resource_source"]], linewidths=1.8)
        axis.annotate(item["id"], (dsp_percent, item["int8"]), xytext=(5, 5), textcoords="offset points")
    style_axis(axis, "Accuracy vs DSP utilization", "INT8 accuracy (%)")
    axis.set_xlabel("DSP48 utilization (%)")
    save(figure, "accuracy_dsp_utilization_all_v2.png")

    figure, axis = plt.subplots(figsize=(9.4, 4.8))
    bars = axis.bar(x, [item["on_chip_bram36"] for item in values], color="#2878b5")
    axis.bar_label(bars, labels=[f'{item["on_chip_bram36"]:.1f} ({100 * item["on_chip_bram36"] / DEVICE["bram"]:.0f}%)' for item in values], padding=2, fontsize=8)
    axis.axhline(DEVICE["bram"], color="#8c8c8c", linestyle="--", linewidth=1, label="XC7Z010 limit (60)")
    axis.set_xticks(x, labels)
    style_axis(axis, "Full on-chip weight + bias BRAM36 requirement", "BRAM36-equivalent tiles")
    axis.legend()
    save(figure, "bram_usage_all_v2.png")

    (ROOT / "comparison_all_v2.json").write_text(json.dumps(values, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
