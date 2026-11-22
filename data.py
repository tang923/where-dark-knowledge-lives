import hashlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def split_indices(targets, seed=661, fraction=0.1):
    targets = torch.as_tensor(targets)
    generator = torch.Generator().manual_seed(seed)
    train, val = [], []
    for label in targets.unique(sorted=True):
        indices = (targets == label).nonzero().flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n = max(1, int(len(indices) * fraction))
        val.extend(indices[:n].tolist())
        train.extend(indices[n:].tolist())
    return train, val


def indices_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def balanced_limit(indices, targets, limit):
    if not limit or len(indices) <= limit:
        return indices
    labels = sorted({int(targets[i]) for i in indices})
    if limit < len(labels):
        raise ValueError("Sample limit must cover every retained class")
    buckets = {label: [] for label in labels}
    for i in indices:
        buckets[int(targets[i])].append(i)
    selected = []
    for row in range(max(map(len, buckets.values()))):
        for label in labels:
            if row < len(buckets[label]):
                selected.append(buckets[label][row])
                if len(selected) == limit:
                    return selected
    return selected


def make_datasets(cfg):
    name = cfg["dataset"]
    if name == "mnist":
        cls = datasets.MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        shift = cfg.get("mnist_shift", 0)
        augmentation = [transforms.RandomCrop(28, padding=shift)] if shift else []
    else:
        cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) if name == "cifar10" else (
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        normalize = transforms.Normalize(*stats)
        augmentation = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    plain = transforms.Compose([transforms.ToTensor(), normalize])
    augmented = transforms.Compose(augmentation + [transforms.ToTensor(), normalize])
    kwargs = {"root": cfg["data_root"], "download": cfg["download"]}
    train_data = cls(train=True, transform=augmented, **kwargs)
    val_data = cls(train=True, transform=plain, **kwargs)
    test_data = cls(train=False, transform=plain, **kwargs)
    train_idx, val_idx = split_indices(train_data.targets, cfg["split_seed"])
    original_train = train_idx[:]
    if cfg["omit"] is not None:
        train_idx = [i for i in train_idx if int(train_data.targets[i]) != cfg["omit"]]
    train_idx = balanced_limit(train_idx, train_data.targets, cfg["train_limit"])
    val_idx = balanced_limit(val_idx, val_data.targets, cfg["eval_limit"])
    test_idx = balanced_limit(list(range(len(test_data))), test_data.targets, cfg["eval_limit"])
    info = {"train_size": len(train_idx), "val_size": len(val_idx), "test_size": len(test_idx),
            "train_hash": indices_hash(train_idx), "val_hash": indices_hash(val_idx),
            "test_hash": indices_hash(test_idx), "full_train_hash": indices_hash(original_train),
            "train_classes": sorted({int(train_data.targets[i]) for i in train_idx})}
    return (Subset(train_data, train_idx), Subset(val_data, val_idx), Subset(test_data, test_idx)), info


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, cfg, *, train=False, epoch=0):
    generator = torch.Generator().manual_seed(cfg["seed"] * 100000 + epoch)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=train,
                      num_workers=cfg["workers"], pin_memory=cfg["device"].startswith("cuda"),
                      worker_init_fn=seed_worker, generator=generator, drop_last=False)
