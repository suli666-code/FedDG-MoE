"""数据模块公开入口：导出 FedDG 数据加载和数据集元信息工具。"""

from .feddg_loaders import (
    build_feddg_dataloaders,
    dataset_domains,
    default_num_classes,
    full_domain_to_code,
    resolve_data_root,
)

__all__ = [
    "build_feddg_dataloaders",
    "dataset_domains",
    "default_num_classes",
    "full_domain_to_code",
    "resolve_data_root",
]
