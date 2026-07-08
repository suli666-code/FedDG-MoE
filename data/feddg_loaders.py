"""Data loading utilities for PACS, OfficeHome, and VLCS FedDG experiments."""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATASET_DIR_NAMES = {
    "pacs": "PACS",
    "officehome": "Office-Home",
    "vlcs": "VLCS",
}

OFFICEHOME_NAME_DICT = {
    "art": "Art",
    "clipart": "Clipart",
    "product": "Product",
    "real_world": "Real World",
}

VLCS_NAME_DICT = {
    "caltech": "CALTECH",
    "labelme": "LABELME",
    "pascal": "PASCAL",
    "sun": "SUN",
}

PACS_NAME_DICT = {
    "p": "photo",
    "a": "art_painting",
    "c": "cartoon",
    "s": "sketch",
}

PACS_SPLIT_DICT = {
    "train": "train",
    "val": "crossval",
    "total": "test",
}

transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class DomainLabelDataset(Dataset):
    def __init__(self, dataset: Dataset, domain_label: int):
        self.dataset = dataset
        self.domain_label = int(domain_label)

    def __getitem__(self, index: int):
        img, label = self.dataset[index]
        return img, label, self.domain_label

    def __len__(self) -> int:
        return len(self.dataset)


class PACSPathDataset(Dataset):
    def __init__(self, imgs: List[str], labels: List[int], domain_label: int, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.domain_label = int(domain_label)
        self.transform = transform

    def __getitem__(self, index: int):
        img = Image.open(self.imgs[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index], self.domain_label

    def __len__(self) -> int:
        return len(self.imgs)


def seed_to_uint32(seed: int) -> int:
    return int(seed) % (2**32)


def seed_worker(worker_id: int) -> None:
    worker_seed = seed_to_uint32(torch.initial_seed())
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loader_generator(seed: int, domain_idx: int, split: str) -> torch.Generator:
    split_offsets = {"train": 0, "val": 10_000, "test": 20_000}
    if split not in split_offsets:
        raise ValueError(f"Unknown dataloader split: {split}")
    generator_seed = int(seed) + int(domain_idx) * 1_000 + split_offsets[split]
    return torch.Generator().manual_seed(generator_seed)


def full_domain_to_code(domain: str) -> str:
    mapping = {domain_name: domain_code for domain_code, domain_name in PACS_NAME_DICT.items()}
    if domain not in mapping:
        raise ValueError(f"Unknown PACS domain name: {domain}. Use one of {list(mapping.keys())}.")
    return mapping[domain]


def dataset_domains(dataset_name: str) -> List[str]:
    if dataset_name == "pacs":
        return ["art_painting", "cartoon", "photo", "sketch"]
    if dataset_name == "officehome":
        return list(OFFICEHOME_NAME_DICT.keys())
    if dataset_name == "vlcs":
        return list(VLCS_NAME_DICT.keys())
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def default_num_classes(dataset_name: str) -> int:
    if dataset_name == "officehome":
        return 65
    if dataset_name == "vlcs":
        return 5
    return 7


def resolve_data_root(dataset_name: str, data_root: str, *, project_dir: str | None = None) -> str:
    dataset_dir_name = DATASET_DIR_NAMES[dataset_name]
    data_root = str(data_root or "").strip()

    if data_root == "":
        if project_dir is None:
            project_dir = os.getcwd()
        datasets_root = os.path.abspath(os.path.join(project_dir, os.pardir, "datasets"))
        resolved = os.path.join(datasets_root, dataset_dir_name)
    else:
        candidate = os.path.abspath(os.path.expanduser(data_root))
        nested = os.path.join(candidate, dataset_dir_name)
        resolved = nested if os.path.isdir(nested) else candidate

    if not os.path.isdir(resolved):
        raise FileNotFoundError(
            f"Cannot find dataset root for --dataset {dataset_name}: {resolved}. "
            f"Use --data_root to point to either the dataset folder or the parent datasets folder."
        )
    return resolved


def build_feddg_dataloaders(
    *,
    dataset_name: str,
    data_root: str,
    target_domain: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    seed: int,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
) -> Tuple[Dict[str, Dict[str, DataLoader]], Dict[str, Dict[str, object]], List[str], str]:
    if dataset_name == "pacs":
        target_domain_code = full_domain_to_code(target_domain)
        dataloaders, datasets_by_domain = _build_pacs_feddg_dataloaders(
            data_root=data_root,
            target_domain_code=target_domain_code,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            seed=seed,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
        )
        return dataloaders, datasets_by_domain, list(PACS_NAME_DICT.keys()), target_domain_code

    domain_map = OFFICEHOME_NAME_DICT if dataset_name == "officehome" else VLCS_NAME_DICT
    dataloaders, datasets_by_domain = _build_imagefolder_feddg_dataloaders(
        root=data_root,
        domain_map=domain_map,
        target_domain=target_domain,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        seed=seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )
    return dataloaders, datasets_by_domain, list(domain_map.keys()), target_domain


def _limit_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()[:max_samples]
    return Subset(dataset, indices)


def _split_subset_indices(length: int, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if length <= 1:
        return list(range(length)), list(range(length))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    train_size = max(1, min(length - 1, int(length * train_ratio)))
    return indices[:train_size], indices[train_size:]


def _find_domain_dir(root: str, domain_dir: str) -> str:
    candidates = [
        os.path.join(root, domain_dir),
        os.path.join(root, domain_dir.replace("_", " ")),
        os.path.join(root, "raw_images", domain_dir),
        os.path.join(root, "raw_images", domain_dir.replace("_", " ")),
        os.path.join(root, "office_home", domain_dir),
        os.path.join(root, "office_home", domain_dir.replace("_", " ")),
        os.path.join(root, "OfficeHomeDataset_10072016", domain_dir),
        os.path.join(root, "OfficeHomeDataset_10072016", domain_dir.replace("_", " ")),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Cannot find domain folder '{domain_dir}' under {root}. Tried: {candidates}"
    )


def _imagefolder_domain_dataset(root: str, domain_dir: str, domain_label: int, split: str, *, seed: int) -> Dataset:
    base_transform = transform_train if split == "train" else transform_test
    domain_path = _find_domain_dir(root, domain_dir)
    split_dirs = {
        "train": ["train"],
        "val": ["crossval", "val"],
        "test": ["test"],
    }
    for split_dir in split_dirs[split]:
        split_path = os.path.join(domain_path, split_dir)
        if os.path.isdir(split_path):
            return DomainLabelDataset(datasets.ImageFolder(split_path, transform=base_transform), domain_label)

    full_for_split = datasets.ImageFolder(domain_path, transform=base_transform)
    if split == "test":
        return DomainLabelDataset(full_for_split, domain_label)
    full_for_indexing = datasets.ImageFolder(domain_path, transform=transform_test)
    train_indices, val_indices = _split_subset_indices(len(full_for_indexing), 0.9, seed)
    selected = train_indices if split == "train" else val_indices
    return DomainLabelDataset(Subset(full_for_split, selected), domain_label)


def _read_pacs_split_file(txt_path: str, raw_images_root: str) -> Tuple[List[str], List[int]]:
    imgs: List[str] = []
    labels: List[int] = []
    with open(txt_path, "r") as f:
        for line_txt in f.readlines():
            image_rel_path, label = line_txt.strip().split(" ")
            imgs.append(os.path.join(raw_images_root, image_rel_path))
            labels.append(int(label) - 1)
    return imgs, labels


def _pacs_domain_dataset(root: str, domain_code: str, domain_label: int, split: str) -> Dataset:
    if domain_code not in PACS_NAME_DICT:
        raise ValueError(f"domain_code must be one of {list(PACS_NAME_DICT.keys())}, got {domain_code}")
    if split not in PACS_SPLIT_DICT:
        raise ValueError(f"split must be one of {list(PACS_SPLIT_DICT.keys())}, got {split}")

    domain_name = PACS_NAME_DICT[domain_code]
    raw_images_root = os.path.join(root, "raw_images")
    split_file = os.path.join(root, "split_files", f"{domain_name}_{PACS_SPLIT_DICT[split]}_kfold.txt")
    imgs, labels = _read_pacs_split_file(split_file, raw_images_root)
    transform = transform_train if split == "train" else transform_test
    return PACSPathDataset(imgs, labels, domain_label, transform)


def _build_imagefolder_feddg_dataloaders(
    *,
    root: str,
    domain_map: Dict[str, str],
    target_domain: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    seed: int,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
) -> Tuple[Dict[str, Dict[str, DataLoader]], Dict[str, Dict[str, object]]]:
    dataloader_dict: Dict[str, Dict[str, DataLoader]] = {}
    dataset_dict: Dict[str, Dict[str, object]] = {}
    domain_list = list(domain_map.keys())
    if target_domain not in domain_list:
        raise ValueError(f"target_domain must be one of {domain_list}, got {target_domain}")

    for domain_idx, domain_key in enumerate(domain_list):
        domain_dir = domain_map[domain_key]
        domain_datasets = {
            "train": _imagefolder_domain_dataset(root, domain_dir, domain_idx, "train", seed=seed),
            "val": _imagefolder_domain_dataset(root, domain_dir, domain_idx, "val", seed=seed),
            "test": _imagefolder_domain_dataset(root, domain_dir, domain_idx, "test", seed=seed),
        }
        domain_datasets["train"] = _limit_dataset(domain_datasets["train"], max_train_samples, seed + domain_idx)
        domain_datasets["val"] = _limit_dataset(domain_datasets["val"], max_eval_samples, seed + domain_idx + 1000)
        domain_datasets["test"] = _limit_dataset(domain_datasets["test"], max_eval_samples, seed + domain_idx + 2000)
        dataset_dict[domain_key] = domain_datasets
        dataloader_dict[domain_key] = _make_domain_loaders(
            domain_datasets=domain_datasets,
            batch_size=batch_size,
            test_batch_size=(test_batch_size if domain_key == target_domain else batch_size),
            num_workers=num_workers,
            seed=seed,
            domain_idx=domain_idx,
        )
    return dataloader_dict, dataset_dict


def _build_pacs_feddg_dataloaders(
    *,
    data_root: str,
    target_domain_code: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int = 4,
    seed: int = 0,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
) -> Tuple[Dict[str, Dict[str, DataLoader]], Dict[str, Dict[str, object]]]:
    dataloader_dict: Dict[str, Dict[str, DataLoader]] = {}
    dataset_dict: Dict[str, Dict[str, object]] = {}
    domain_list: List[str] = list(PACS_NAME_DICT.keys())
    if target_domain_code not in domain_list:
        raise ValueError(f"target_domain_code must be one of {domain_list}, got {target_domain_code}")

    for domain_idx, domain_code in enumerate(domain_list):
        domain_datasets = {
            "train": _pacs_domain_dataset(data_root, domain_code, domain_idx, "train"),
            "val": _pacs_domain_dataset(data_root, domain_code, domain_idx, "val"),
            "test": _pacs_domain_dataset(data_root, domain_code, domain_idx, "total"),
        }
        domain_datasets["train"] = _limit_dataset(domain_datasets["train"], max_train_samples, seed + domain_idx)
        domain_datasets["val"] = _limit_dataset(domain_datasets["val"], max_eval_samples, seed + domain_idx + 1000)
        domain_datasets["test"] = _limit_dataset(domain_datasets["test"], max_eval_samples, seed + domain_idx + 2000)
        dataset_dict[domain_code] = domain_datasets
        dataloader_dict[domain_code] = _make_domain_loaders(
            domain_datasets=domain_datasets,
            batch_size=batch_size,
            test_batch_size=(test_batch_size if domain_code == target_domain_code else batch_size),
            num_workers=num_workers,
            seed=seed,
            domain_idx=domain_idx,
        )
    return dataloader_dict, dataset_dict


def _make_domain_loaders(
    *,
    domain_datasets: Dict[str, Dataset],
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    seed: int,
    domain_idx: int,
) -> Dict[str, DataLoader]:
    return {
        "train": DataLoader(
            domain_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=make_loader_generator(seed, domain_idx, "train"),
        ),
        "val": DataLoader(
            domain_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=make_loader_generator(seed, domain_idx, "val"),
        ),
        "test": DataLoader(
            domain_datasets["test"],
            batch_size=test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=make_loader_generator(seed, domain_idx, "test"),
        ),
    }
