from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import make_loaders, make_test_loader, set_seed
from .per_layer_ptq import IntegerCNN, build_parameters, calibrate, export


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_module(design_dir: Path):
    module_name = f"architecture_{design_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, design_dir / "model.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load model.py from {design_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": state}


def _resolve(design_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (design_dir / path).resolve()


def _default_checkpoint(design_dir: Path, config: dict[str, Any]) -> Path:
    return _resolve(design_dir, config["paths"]["checkpoint"])


def _checkpoint_provenance(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    """Describe whether the loaded weights originate from training or initialization."""
    checkpoint_metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    accuracy_source = checkpoint_metrics.get("accuracy_source", "")
    if accuracy_source.startswith("measured_"):
        provenance = {
            "evaluation_status": "trained_checkpoint",
            "weights_provenance": "trained CIFAR-10 checkpoint",
            "checkpoint": str(checkpoint_path),
            "checkpoint_accuracy_source": accuracy_source,
        }
        for key in ("fp32_test_accuracy", "best_val_accuracy", "epochs", "seed"):
            if key in checkpoint_metrics:
                provenance[f"checkpoint_{key}"] = checkpoint_metrics[key]
        return provenance
    return {
        "evaluation_status": "not_evaluated",
        "weights_provenance": "deterministic initialization; structural RTL vectors only",
        "checkpoint": str(checkpoint_path),
        "checkpoint_accuracy_source": accuracy_source or "not_measured_deterministic_initialization",
    }


def _evaluate_fp32(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += float(criterion(logits, labels).item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.numel()
    return loss_sum / total, correct / total


def _evaluate_int8(model: IntegerCNN, loader, device: torch.device, max_samples: int = 0) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            if max_samples > 0 and total >= max_samples:
                break
            if max_samples > 0 and total + len(labels) > max_samples:
                keep = max_samples - total
                images = images[:keep]
                labels = labels[:keep]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model.forward(images)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.numel()
    return correct / total


def _augment_batch(images: torch.Tensor) -> torch.Tensor:
    """CIFAR random crop/flip on a batch / 在 batch 上执行 CIFAR 随机裁剪与翻转。"""
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    tops = torch.randint(0, 9, (len(images),), device=images.device)
    lefts = torch.randint(0, 9, (len(images),), device=images.device)
    row_index = tops[:, None] + torch.arange(32, device=images.device)[None, :]
    rows = padded.gather(2, row_index[:, None, :, None].expand(-1, images.shape[1], -1, 40))
    col_index = lefts[:, None] + torch.arange(32, device=images.device)[None, :]
    cropped = rows.gather(3, col_index[:, None, None, :].expand(-1, images.shape[1], 32, -1))
    flip = torch.rand(len(images), device=images.device) < 0.5
    cropped[flip] = torch.flip(cropped[flip], dims=(-1,))
    return cropped


def _model_statistics(model: nn.Module) -> tuple[int, int]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    macs = 0
    hooks = []

    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            output_elements = output.shape[1] * output.shape[2] * output.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            macs += int(output_elements * kernel_ops)
        elif isinstance(module, nn.Linear):
            macs += int(module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    model.eval()
    try:
        statistics_device = next(model.parameters()).device
    except StopIteration:
        statistics_device = torch.device("cpu")
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32, device=statistics_device))
    for handle in hooks:
        handle.remove()
    return parameters, macs


def _train(args, design_dir: Path, config: dict[str, Any], model: nn.Module, device: torch.device) -> None:
    training = config["training"]
    epochs = args.epochs or int(training["epochs"])
    batch_size = args.batch_size or int(training["batch_size"])
    data_root = args.data_root or _resolve(design_dir, config["paths"]["data_root"])
    train_loader, val_loader, test_loader = make_loaders(
        data_root,
        int(training["val_size"]),
        batch_size,
        args.num_workers,
        fast_train=True,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["learning_rate"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            images = _augment_batch(images)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            loss_sum += float(loss.item()) * labels.numel()
        val_loss, val_acc = _evaluate_fp32(model, val_loader, device)
        scheduler.step()
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:03d}/{epochs:03d} "
            f"train_loss={loss_sum / total:.4f} train_acc={correct / total:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
    model.load_state_dict(best_state)
    test_loss, test_acc = _evaluate_fp32(model, test_loader, device)
    training_path = config["paths"].get("training_checkpoint", config["paths"]["checkpoint"])
    checkpoint_path = args.checkpoint or _resolve(design_dir, training_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "architecture_revision": config.get("architecture_revision", "unspecified"),
        "accuracy_source": "measured_fp32_test",
        "parameter_count": _model_statistics(model)[0],
        "macs_per_image": _model_statistics(model)[1],
        "best_val_accuracy": best_val,
        "fp32_test_accuracy": test_acc,
        "fp32_test_loss": test_loss,
        "training_seconds": time.time() - started,
        "epochs": epochs,
        "seed": int(config["training"]["seed"]),
    }
    torch.save(
        {
            "design_id": config["design_id"],
            "model_state_dict": best_state,
            "metrics": metrics,
            "config": config,
        },
        checkpoint_path,
    )
    metrics_path = checkpoint_path.parent.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"checkpoint={checkpoint_path}")
    print(json.dumps(metrics, indent=2))


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])


def _prepare_ptq(model: nn.Module, config: dict[str, Any], design_dir: Path, device: torch.device,
                  data_root: Path, output_dir: Path, calibration_samples: int = 2048,
                  percentile: float = 99.9, provenance: dict[str, Any] | None = None) -> IntegerCNN:
    """Calibrate, export and return the shared integer golden model."""
    _, calibration_loader, _ = make_loaders(
        data_root, int(config["training"]["val_size"]),
        int(config["evaluation"]["batch_size"]), 0)
    calibration = calibrate(model, calibration_loader, config["topology"], device,
                            calibration_samples, percentile)
    arrays, layers = build_parameters(model, config["topology"], calibration)
    provenance = provenance or {
        "evaluation_status": config.get("evaluation_status", "not_evaluated"),
        "weights_provenance": "deterministic initialization; structural RTL vectors only",
    }
    quantization = {
        "scheme": "per-layer symmetric INT8 PTQ",
        "architecture_revision": config.get("architecture_revision", "unspecified"),
        "evaluation_status": provenance["evaluation_status"],
        "weights_provenance": provenance["weights_provenance"],
        "calibration": {
            "source": "CIFAR-10 validation split", "samples": calibration["samples"],
            "method": "sampled absolute percentile", "activation_percentile": percentile,
        },
        "activation_maxima": calibration["maxima"],
        "activation_thresholds": calibration["thresholds"],
        "activation_scales": calibration["scales"],
        "layers": layers,
    }
    for key, value in provenance.items():
        if key not in quantization:
            quantization[key] = value
    export(output_dir, arrays, quantization)
    return IntegerCNN(arrays, quantization, config["topology"], device)


def _write_accuracy_artifacts(design_dir: Path, fp32: float, int8: float,
                              parameter_count: int, macs_per_image: int) -> None:
    """Write machine-readable metrics and a compact accuracy chart."""
    artifacts = design_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts / "metrics.json"
    metrics = _load_json(metrics_path) if metrics_path.exists() else {}
    metrics.update({"fp32_test_accuracy": fp32, "int8_test_accuracy": int8,
                    "int8_accuracy_drop": fp32 - int8,
                    "accuracy_source": "measured_fp32_and_int8_ptq_test",
                    "parameter_count": parameter_count,
                    "macs_per_image": macs_per_image})
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(5.6, 3.6))
        bars = axis.bar(["FP32", "INT8 PTQ"], [100 * fp32, 100 * int8], color=["#2878b5", "#d9534f"])
        axis.set_ylim(0, 100)
        axis.set_ylabel("CIFAR-10 test accuracy (%)")
        axis.set_title(f"{design_dir.name}: FP32 vs INT8")
        axis.bar_label(bars, fmt="%.2f%%", padding=3)
        figure.tight_layout()
        figure.savefig(artifacts / "accuracy_comparison.png", dpi=180)
        plt.close(figure)
    except ImportError:
        print("warning: matplotlib unavailable; JSON metrics were still written")


def run_design(design_dir: Path) -> None:
    design_dir = design_dir.resolve()
    config = _load_json(design_dir / "config.json")
    model_module = _load_model_module(design_dir)
    parser = argparse.ArgumentParser(description=f"{config['design_id']} {config['name']} experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary")
    initialize_parser = subparsers.add_parser("initialize-checkpoint")
    _add_common_options(initialize_parser)
    train_parser = subparsers.add_parser("train")
    _add_common_options(train_parser)
    train_parser.add_argument("--epochs", type=int)

    fp32_parser = subparsers.add_parser("evaluate-fp32")
    _add_common_options(fp32_parser)
    int8_parser = subparsers.add_parser("evaluate-int8")
    _add_common_options(int8_parser)
    int8_parser.add_argument("--max-samples", type=int, default=0)

    compare_parser = subparsers.add_parser("compare-accuracy")
    _add_common_options(compare_parser)
    compare_parser.add_argument("--calibration-samples", type=int, default=2048)
    compare_parser.add_argument("--activation-percentile", type=float, default=99.9)

    export_parser = subparsers.add_parser("export-int8")
    _add_common_options(export_parser)
    export_parser.add_argument("--output-dir", type=Path)
    export_parser.add_argument("--calibration-samples", type=int, default=2048)
    export_parser.add_argument("--activation-percentile", type=float, default=99.9)

    vector_parser = subparsers.add_parser("generate-rtl-vectors")
    _add_common_options(vector_parser)
    vector_parser.add_argument("--num-samples", type=int, default=1)
    vector_parser.add_argument("--output-dir", type=Path)

    subparsers.add_parser("rtl-test")
    # 中文：兼容位置子命令和早期文档中的 --command 写法；无参数时显示网络摘要。
    # English: Accept positional and legacy --command syntax; default to summary.
    argv = sys.argv[1:]
    if not argv:
        argv = ["summary"]
    elif argv[0] == "--command":
        if len(argv) < 2:
            parser.error("--command requires a command name")
        argv = [argv[1], *argv[2:]]
    elif argv[0].startswith("--command="):
        argv = [argv[0].split("=", 1)[1], *argv[1:]]
    args = parser.parse_args(argv)
    set_seed(int(config["training"]["seed"]))
    model = model_module.build_model()
    parameter_count, macs_per_image = _model_statistics(model)

    if args.command == "summary":
        print(json.dumps({"design_id": config["design_id"], "parameter_count": parameter_count,
                          "macs_per_image": macs_per_image}, indent=2))
        return

    if args.command == "initialize-checkpoint":
        checkpoint_path = args.checkpoint or _default_checkpoint(design_dir, config)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "architecture_revision": config.get("architecture_revision", "unspecified"),
            "parameter_count": parameter_count,
            "macs_per_image": macs_per_image,
            "accuracy_source": "not_measured_deterministic_initialization",
            "evaluation_status": config.get("evaluation_status", "not_evaluated"),
        }
        torch.save(
            {
                "design_id": config["design_id"],
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "config": config,
            },
            checkpoint_path,
        )
        print(json.dumps({"checkpoint": str(checkpoint_path), **metrics}, indent=2))
        return

    if args.command == "rtl-test":
        status = _load_json(design_dir / "rtl" / "status.json")
        if status.get("implementation_status") != "verilator_verified":
            print(json.dumps(status, indent=2))
            raise SystemExit(2)
        script = status.get("test_script")
        if script:
            # PowerShell-launched Cygwin does not reliably inherit the current directory.
            # PowerShell 启动 Cygwin 时工作目录不可靠，因此显式转换路径并执行 cd。
            if os.name == "nt":
                drive = design_dir.drive[0].lower()
                tail = design_dir.as_posix()[3:]
                cygwin_dir = f"/cygdrive/{drive}/{tail}"
                command = [r"C:\cygwin64\bin\bash.exe", "-lc", f"cd '{cygwin_dir}' && bash '{script}'"]
            else:
                command = ["bash", script]
            raise SystemExit(subprocess.run(command, cwd=design_dir, shell=False, check=False).returncode)
        command = status.get("test_command")
        if not command:
            raise RuntimeError("verified RTL status is missing test_command")
        raise SystemExit(subprocess.run(command, cwd=design_dir, shell=True, check=False).returncode)

    device = _device(args.device)
    seed = int(config["training"]["seed"])
    set_seed(seed)
    if args.command == "train":
        _train(args, design_dir, config, model, device)
        return

    checkpoint_path = args.checkpoint or _default_checkpoint(design_dir, config)
    checkpoint = _load_checkpoint(model, checkpoint_path, device)
    provenance = _checkpoint_provenance(checkpoint, checkpoint_path)
    model.to(device).eval()
    if args.command == "export-int8":
        output = args.output_dir or _resolve(design_dir, config["paths"]["int8_export"])
        data_root = args.data_root or _resolve(design_dir, config["paths"]["data_root"])
        _prepare_ptq(model, config, design_dir, device, data_root, output,
                     args.calibration_samples, args.activation_percentile, provenance)
        print(f"exported={output}")
        return

    data_root = args.data_root or _resolve(design_dir, config["paths"]["data_root"])
    batch_size = args.batch_size or int(config["evaluation"]["batch_size"])
    loader = make_test_loader(data_root, batch_size, args.num_workers)
    if args.command == "evaluate-fp32":
        loss, accuracy = _evaluate_fp32(model, loader, device)
        print(json.dumps({"fp32_test_loss": loss, "fp32_test_accuracy": accuracy}, indent=2))
        return
    if args.command == "evaluate-int8":
        output = _resolve(design_dir, config["paths"]["int8_export"])
        integer_model = _prepare_ptq(model, config, design_dir, device, data_root, output,
                                     provenance=provenance)
        accuracy = _evaluate_int8(integer_model, loader, device, args.max_samples)
        print(json.dumps({"int8_test_accuracy": accuracy, "samples": args.max_samples or 10000}, indent=2))
        return
    if args.command == "compare-accuracy":
        loss, fp32_accuracy = _evaluate_fp32(model, loader, device)
        output = _resolve(design_dir, config["paths"]["int8_export"])
        integer_model = _prepare_ptq(model, config, design_dir, device, data_root, output,
                                     args.calibration_samples, args.activation_percentile, provenance)
        int8_accuracy = _evaluate_int8(integer_model, loader, device)
        _write_accuracy_artifacts(design_dir, fp32_accuracy, int8_accuracy,
                                  parameter_count, macs_per_image)
        print(json.dumps({"fp32_test_loss": loss, "fp32_test_accuracy": fp32_accuracy,
                          "int8_test_accuracy": int8_accuracy,
                          "accuracy_drop": fp32_accuracy - int8_accuracy,
                          "parameter_count": parameter_count,
                          "macs_per_image": macs_per_image}, indent=2))
        return
    if args.command == "generate-rtl-vectors":
        output = args.output_dir or (design_dir / "rtl" / "sim" / "data")
        output.mkdir(parents=True, exist_ok=True)
        images, labels = next(iter(make_test_loader(data_root, args.num_samples, 0)))
        images = images.to(device)
        int8_output = _resolve(design_dir, config["paths"]["int8_export"])
        integer_model = _prepare_ptq(model, config, design_dir, device, data_root, int8_output,
                                     provenance=provenance)
        inputs_q = integer_model.quantize_input(images)
        with torch.no_grad():
            logits = integer_model.forward_quantized(inputs_q)
        input_lines = [" ".join(str(int(value)) for value in sample.reshape(-1).cpu()) for sample in inputs_q]
        golden_lines = []
        for label, sample_logits in zip(labels, logits):
            predicted = int(sample_logits.argmax().item())
            max_logit = int(sample_logits[predicted].item())
            values = [int(label), predicted, max_logit, *[int(value) for value in sample_logits.cpu()]]
            golden_lines.append(" ".join(str(value) for value in values))
        (output / "input_int8.txt").write_text("\n".join(input_lines) + "\n", encoding="ascii")
        (output / "golden_logits.txt").write_text("\n".join(golden_lines) + "\n", encoding="ascii")
        metadata = {
            "design_id": config["design_id"],
            "samples": len(labels),
            "quantization_file": str(int8_output / "quantization.json"),
            "architecture_revision": config.get("architecture_revision", "unspecified"),
            **provenance,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"vectors={output}")
