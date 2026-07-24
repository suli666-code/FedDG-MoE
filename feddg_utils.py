"""Training utilities for seeding, logging, checkpoints, state I/O, and aggregation."""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from client_train import (
    _infer_adapter_mode,
    _is_adapter_param_name,
    setup_trainable_params,
)


@torch.no_grad()
def aggregate_main_expert(
    client_adapter_states: List[Dict[str, torch.Tensor]],
    client_classifier_states: List[Dict[str, torch.Tensor]],
    client_prototype_states: List[torch.Tensor],
    client_sample_counts: List[float],
    client_prototype_counts: List[torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """FedAvg for supplied adapter state, classifier, and class prototypes."""
    if not client_adapter_states:
        raise ValueError("Cannot aggregate empty client states.")
    sample_count_tensor = torch.tensor(client_sample_counts, dtype=torch.float32)
    if float(sample_count_tensor.sum().item()) <= 0.0:
        sample_weights = torch.full_like(sample_count_tensor, 1.0 / float(len(client_adapter_states)))
    else:
        sample_weights = sample_count_tensor / sample_count_tensor.sum()

    avg_adapter: Dict[str, torch.Tensor] = {}
    example_adapter_state = client_adapter_states[0]
    for key in example_adapter_state.keys():
        stacked = torch.stack([state[key].to(torch.float32) for state in client_adapter_states], dim=0)
        view_shape = (len(client_adapter_states),) + (1,) * (stacked.dim() - 1)
        avg_adapter[key] = (stacked * sample_weights.view(view_shape)).sum(dim=0)

    avg_classifier: Dict[str, torch.Tensor] = {}
    example_classifier_state = client_classifier_states[0]
    for key in example_classifier_state.keys():
        stacked = torch.stack([state[key].to(torch.float32) for state in client_classifier_states], dim=0)
        view_shape = (len(client_classifier_states),) + (1,) * (stacked.dim() - 1)
        avg_classifier[key] = (stacked * sample_weights.view(view_shape)).sum(dim=0)

    proto_stack = torch.stack([state.to(torch.float32) for state in client_prototype_states], dim=0)
    count_stack = torch.stack([counts.to(torch.float32) for counts in client_prototype_counts], dim=0)
    class_totals = count_stack.sum(dim=0)
    avg_prototypes = (proto_stack * count_stack.unsqueeze(-1)).sum(dim=0) / class_totals.clamp_min(1.0).unsqueeze(-1)
    missing_class_mask = class_totals <= 0
    if bool(missing_class_mask.any()):
        view_shape = (len(client_prototype_states),) + (1,) * (proto_stack.dim() - 1)
        fallback_proto = (proto_stack * sample_weights.view(view_shape)).sum(dim=0)
        avg_prototypes[missing_class_mask] = fallback_proto[missing_class_mask]
    valid_class_mask = class_totals > 0
    if bool(valid_class_mask.any()):
        avg_prototypes[valid_class_mask] = torch.nn.functional.normalize(
            avg_prototypes[valid_class_mask],
            dim=-1,
            eps=1e-6,
        )
    return avg_adapter, avg_classifier, avg_prototypes


def setup_run_logger(*, logger_name: str, log_path: str) -> logging.Logger:
    level = logging.INFO
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda" in msg and "oom" in msg)


def save_checkpoint(*, checkpoint_path: str, payload: Dict[str, Any], logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(payload, checkpoint_path)
    logger.warning(f"[Checkpoint] saved to: {checkpoint_path}")


def resolve_num_workers(user_value: int) -> int:
    if user_value >= 0:
        return user_value
    cpu_count = os.cpu_count() or 8
    return max(2, min(8, cpu_count // 2))


def seed_to_uint32(seed: int) -> int:
    return int(seed) % (2**32)


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed_to_uint32(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def extract_adapter_state_from_model(
    model: torch.nn.Module,
    *,
    include_style_params: bool = False,
) -> Dict[str, torch.Tensor]:
    """Extract trainable adapter state, optionally including style branches for FedAvg."""
    adapter_state: Dict[str, torch.Tensor] = {}
    featurizer_state = model[0].state_dict()
    for key, value in featurizer_state.items():
        is_style_param = ("style_down" in key) or ("style_up" in key)
        if _is_adapter_param_name(key) and (include_style_params or not is_style_param):
            adapter_state[key] = value.detach().cpu().to(torch.float32)
    if len(adapter_state) == 0:
        raise RuntimeError(
            "Failed to extract adapter weights. "
            "Check if '_is_adapter_param_name' matches your layer names."
        )
    return adapter_state


def extract_classifier_state_from_model(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    classifier_state: Dict[str, torch.Tensor] = {}
    for key, value in model[1].state_dict().items():
        classifier_state[key] = value.detach().cpu().to(torch.float32)
    if len(classifier_state) == 0:
        raise RuntimeError("No classifier parameters found in global model state_dict.")
    return classifier_state


def count_params(params) -> int:
    return int(sum(param.numel() for param in params))


def log_model_param_stats(model: torch.nn.Module, logger: logging.Logger) -> None:
    featurizer = model[0]
    classifier = model[1]

    total_params = count_params(model.parameters())
    grad_flags = [param.requires_grad for param in model.parameters()]
    adapter_mode = _infer_adapter_mode(featurizer)
    setup_trainable_params(model, adapter_mode)
    trainable_params = count_params(param for param in model.parameters() if param.requires_grad)
    for param, flag in zip(model.parameters(), grad_flags):
        param.requires_grad = flag

    adapter_params = count_params(
        param for name, param in featurizer.named_parameters() if _is_adapter_param_name(name)
    )
    classifier_params = count_params(classifier.parameters())

    logger.info(
        "[Params] total=%d trainable=%d adapter=%d classifier=%d",
        total_params,
        trainable_params,
        adapter_params,
        classifier_params,
    )


@torch.no_grad()
def load_adapter_state_to_model(
    model: torch.nn.Module, adapter_state: Dict[str, torch.Tensor]
) -> Tuple[int, int, List[str]]:
    state_refs = model.state_dict(keep_vars=True)
    matched = 0
    unmatched_keys: List[str] = []
    for key, value in adapter_state.items():
        candidates = [key, f"0.{key}"]
        target_key = next((cand for cand in candidates if cand in state_refs), None)
        if target_key is None:
            unmatched_keys.append(key)
            continue
        target = state_refs[target_key]
        if value.shape != target.shape:
            unmatched_keys.append(key)
            continue
        target.copy_(value.to(device=target.device, dtype=target.dtype))
        matched += 1
    if matched == 0:
        raise RuntimeError("No adapter keys matched model state_dict keys.")
    return matched, len(adapter_state), unmatched_keys


@torch.no_grad()
def load_classifier_state_to_model(model: torch.nn.Module, classifier_state: Dict[str, torch.Tensor]) -> None:
    state_refs = model[1].state_dict(keep_vars=True)
    for key, value in classifier_state.items():
        if key not in state_refs:
            raise KeyError(f"Classifier key '{key}' not found.")
        target = state_refs[key]
        if value.shape != target.shape:
            raise ValueError(
                f"Classifier shape mismatch for '{key}': "
                f"{tuple(value.shape)} vs {tuple(target.shape)}"
            )
        target.copy_(value.to(device=target.device, dtype=target.dtype))
