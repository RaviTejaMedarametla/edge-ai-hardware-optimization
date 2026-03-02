from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATASETS = {
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
}


def build_loaders(dataset_name: str, batch_size: int, train_subset: int | None, val_subset: int | None) -> tuple[DataLoader, DataLoader]:
    if dataset_name not in DATASETS:
        msg = f"Unsupported dataset '{dataset_name}'. Use one of: {list(DATASETS)}"
        raise ValueError(msg)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds_cls = DATASETS[dataset_name]

    train_ds = ds_cls(root="data", train=True, download=True, transform=transform)
    val_ds = ds_cls(root="data", train=False, download=True, transform=transform)

    if train_subset is not None:
        train_ds = Subset(train_ds, list(range(min(train_subset, len(train_ds)))))
    if val_subset is not None:
        val_ds = Subset(val_ds, list(range(min(val_subset, len(val_ds)))))

    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader
