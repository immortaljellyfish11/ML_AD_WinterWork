from __future__ import annotations

import argparse
import copy
import os
import pickle
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fixed_point_simulator import export_quantized_parameters


# CIFAR-10 常用归一化参数。训练和定点仿真阶段必须保持一致，否则精度会明显波动。
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def set_seed(seed: int) -> None:
	"""固定随机种子，尽量保证训练过程可复现。"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def load_cifar10_batch(batch_path: Path) -> Tuple[np.ndarray, np.ndarray]:
	"""读取一个 CIFAR-10 数据批次，返回原始图像和标签。"""
	with batch_path.open("rb") as handle:
		batch = pickle.load(handle, encoding="latin1")

	data_key = b"data" if b"data" in batch else "data"
	labels_key = b"labels" if b"labels" in batch else "labels"
	images = batch[data_key].reshape(-1, 3, 32, 32).astype(np.uint8)
	labels = np.asarray(batch[labels_key], dtype=np.int64)
	return images, labels


def load_cifar10_dataset(root: Path, val_size: int = 5000):
	"""从本地 CIFAR-10 原始文件中读取训练集、验证集和测试集。"""
	train_images_list = []
	train_labels_list = []

	for index in range(1, 6):
		batch_images, batch_labels = load_cifar10_batch(root / f"data_batch_{index}")
		train_images_list.append(batch_images)
		train_labels_list.append(batch_labels)

	train_images = np.concatenate(train_images_list, axis=0)
	train_labels = np.concatenate(train_labels_list, axis=0)
	test_images, test_labels = load_cifar10_batch(root / "test_batch")

	if val_size <= 0 or val_size >= len(train_images):
		raise ValueError("val_size 必须大于 0 且小于训练集样本数。")

	split_index = len(train_images) - val_size
	return (
		train_images[:split_index],
		train_labels[:split_index],
		train_images[split_index:],
		train_labels[split_index:],
		test_images,
		test_labels,
	)


class CIFAR10ArrayDataset(Dataset):
	"""基于 NumPy 数组的 CIFAR-10 数据集封装，便于直接交给 DataLoader。"""

	def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = False):
		self.images = images
		self.labels = labels
		self.train = train

	def __len__(self) -> int:
		return len(self.images)

	def _train_augment(self, image: np.ndarray) -> np.ndarray:
		"""仅在训练阶段做最基础的数据增强，帮助模型更稳健。"""
		# CIFAR-10 图像很小，最常见的增强是先做 4 像素填充，再随机裁剪回 32x32。
		padded = np.pad(image, ((0, 0), (4, 4), (4, 4)), mode="reflect")
		top = np.random.randint(0, 9)
		left = np.random.randint(0, 9)
		cropped = padded[:, top : top + 32, left : left + 32]

		# 30% 概率随机水平翻转。
		if np.random.rand() < 0.3:
			cropped = cropped[:, :, ::-1]

		return cropped

	def __getitem__(self, index: int):
		image = self.images[index].copy()
		label = int(self.labels[index])

		if self.train:
			image = self._train_augment(image)

		# 归一化到 [0, 1]，再使用 CIFAR-10 的均值方差做标准化。
		image = image.astype(np.float32) / 255.0
		image = (image - CIFAR10_MEAN[:, None, None]) / CIFAR10_STD[:, None, None]

		# PyTorch 需要的是 float32 Tensor 和 long 类型标签。
		return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


class CIFAR10CNN(nn.Module):
	"""一个比较轻量但足够做 CIFAR-10 分类的卷积网络。"""

	def __init__(self) -> None:
		super().__init__()

		# 这里故意不使用 BatchNorm，原因是：后续定点仿真时要处理的算子更少，便于手工实现。
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
		)
		self.classifier = nn.Linear(128, 10)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		outputs = self.features(inputs)
		outputs = torch.flatten(outputs, 1)
		return self.classifier(outputs)


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
) -> Tuple[float, float]:
	"""完成一个训练轮次，并返回平均损失与准确率。"""
	model.train()
	running_loss = 0.0
	running_correct = 0
	running_total = 0

	for images, labels in loader:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)
		logits = model(images)
		loss = criterion(logits, labels)
		loss.backward()
		optimizer.step()

		batch_size = labels.size(0)
		running_loss += float(loss.item()) * batch_size
		running_correct += int((logits.argmax(dim=1) == labels).sum().item())
		running_total += batch_size

	return running_loss / running_total, running_correct / running_total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
	"""在验证集或测试集上评估模型。"""
	model.eval()
	running_loss = 0.0
	running_correct = 0
	running_total = 0

	for images, labels in loader:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		logits = model(images)
		loss = criterion(logits, labels)

		batch_size = labels.size(0)
		running_loss += float(loss.item()) * batch_size
		running_correct += int((logits.argmax(dim=1) == labels).sum().item())
		running_total += batch_size

	return running_loss / running_total, running_correct / running_total


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="训练 CIFAR-10 神经网络并导出 32bit 定点参数")
	parser.add_argument(
		"--data-root",
		type=Path,
		default=Path(__file__).resolve().parent / "cifar-10-batches-py",
		help="CIFAR-10 原始数据所在目录",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "outputs",
		help="模型和定点参数保存目录",
	)
	parser.add_argument("--epochs", type=int, default=25, help="训练轮数")
	parser.add_argument("--batch-size", type=int, default=128, help="训练批大小")
	parser.add_argument("--lr", type=float, default=0.01, help="初始学习率")
	parser.add_argument("--val-size", type=int, default=7500, help="从训练集切出的验证集大小")
	parser.add_argument("--seed", type=int, default=9, help="随机种子")
	parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数，Windows 下建议保持 0")
	parser.add_argument("--frac-bits", type=int, default=16, help="导出定点参数时使用的小数位数")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	if not args.data_root.exists():
		raise FileNotFoundError(f"找不到 CIFAR-10 数据目录: {args.data_root}")

	args.output_dir.mkdir(parents=True, exist_ok=True)

	train_images, train_labels, val_images, val_labels, test_images, test_labels = load_cifar10_dataset(
		args.data_root, val_size=args.val_size
	)

	train_dataset = CIFAR10ArrayDataset(train_images, train_labels, train=True)
	val_dataset = CIFAR10ArrayDataset(val_images, val_labels, train=False)
	test_dataset = CIFAR10ArrayDataset(test_images, test_labels, train=False)

	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = CIFAR10CNN().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

	best_state = copy.deepcopy(model.state_dict())
	best_val_acc = 0.0

	print(f"训练设备: {device}")
	print(f"训练集样本数: {len(train_dataset)}，验证集样本数: {len(val_dataset)}，测试集样本数: {len(test_dataset)}")

	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		scheduler.step()

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_state = copy.deepcopy(model.state_dict())

		current_lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch:03d}/{args.epochs:03d} | "
			f"lr={current_lr:.6f} | "
			f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
			f"val loss={val_loss:.4f}, acc={val_acc:.4f}"
		)

	# 训练完成后，切回验证集上表现最好的参数再做最终测试。
	model.load_state_dict(best_state)
	test_loss, test_acc = evaluate(model, test_loader, criterion, device)
	serializable_args = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}

	float_ckpt_path = args.output_dir / "cifar10_cnn_best.pth"
	torch.save(
		{
			"model_state_dict": best_state,
			"best_val_acc": best_val_acc,
			"test_acc": test_acc,
			"test_loss": test_loss,
			"args": serializable_args,
		},
		float_ckpt_path,
	)

	fixed_point_path = args.output_dir / f"cifar10_cnn_q{args.frac_bits}.npz"
	export_quantized_parameters(best_state, fixed_point_path, frac_bits=args.frac_bits)

	print(f"最佳验证集准确率: {best_val_acc:.4f}")
	print(f"测试集准确率: {test_acc:.4f}")
	print(f"浮点模型已保存到: {float_ckpt_path}")
	print(f"32bit 定点参数已保存到: {fixed_point_path}")


if __name__ == "__main__":
	main()
