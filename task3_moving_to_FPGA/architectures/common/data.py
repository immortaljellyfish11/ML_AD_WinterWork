from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    data_key = b"data" if b"data" in batch else "data"
    labels_key = b"labels" if b"labels" in batch else "labels"
    images = np.asarray(batch[data_key], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(batch[labels_key], dtype=np.int64)
    return images, labels


def load_splits(root: Path, val_size: int) -> tuple[np.ndarray, ...]:
    train_parts = [load_batch(root / f"data_batch_{index}") for index in range(1, 6)]
    train_images = np.concatenate([part[0] for part in train_parts], axis=0)
    train_labels = np.concatenate([part[1] for part in train_parts], axis=0)
    test_images, test_labels = load_batch(root / "test_batch")
    if not 0 < val_size < len(train_images):
        raise ValueError("val_size must be between 1 and 49,999")
    split = len(train_images) - val_size
    return (
        train_images[:split],
        train_labels[:split],
        train_images[split:],
        train_labels[split:],
        test_images,
        test_labels,
    )


class CIFAR10Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool, precompute: bool = False) -> None:
        self.images = images
        self.labels = labels
        self.train = train
        # Fast training mode normalizes once; augmentation is then batched on the training device.
        # 快速训练模式只归一化一次，数据增强随后在训练设备上按 batch 完成。
        self.normalized = None
        if precompute:
            normalized = images.astype(np.float32) / 255.0
            self.normalized = (normalized - CIFAR10_MEAN[None, :, None, None]) / CIFAR10_STD[None, :, None, None]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalized is not None:
            return torch.from_numpy(self.normalized[index]), torch.tensor(int(self.labels[index]), dtype=torch.long)
        image = self.images[index].copy()
        if self.train:
            padded = np.pad(image, ((0, 0), (4, 4), (4, 4)), mode="reflect")
            top = np.random.randint(0, 9)
            left = np.random.randint(0, 9)
            image = padded[:, top : top + 32, left : left + 32]
            if np.random.rand() < 0.5:
                image = image[:, :, ::-1]
        image = image.astype(np.float32) / 255.0
        image = (image - CIFAR10_MEAN[:, None, None]) / CIFAR10_STD[:, None, None]
        return torch.from_numpy(image.copy()), torch.tensor(int(self.labels[index]), dtype=torch.long)


def make_loaders(
    root: Path,
    val_size: int,
    batch_size: int,
    num_workers: int,
    fast_train: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    splits = load_splits(root, val_size)
    train = CIFAR10Dataset(splits[0], splits[1], train=not fast_train, precompute=fast_train)
    val = CIFAR10Dataset(splits[2], splits[3], train=False)
    test = CIFAR10Dataset(splits[4], splits[5], train=False)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train, shuffle=True, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )


def make_test_loader(root: Path, batch_size: int, num_workers: int = 0) -> DataLoader:
    images, labels = load_batch(root / "test_batch")
    dataset = CIFAR10Dataset(images, labels, train=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
