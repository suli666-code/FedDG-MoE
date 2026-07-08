# Evaluation utilities.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader

from client_train import (
    STYLE_GROUP_NAMES,
    collect_style_anchor_feature_groups,
    get_style_group_adapter_names,
)
from feddg_utils import load_adapter_state_to_model
from network.get_network import set_adapter_mode


def _tensor_report(sources: List[str], values: torch.Tensor) -> Dict[str, float]:
    return {src: float(values[idx].detach().cpu().item()) for idx, src in enumerate(sources)}


def _group_for_style_state_key(
    key: str,
    group_adapter_names: dict[str, tuple[str, ...]],
) -> str:
    if (".adapter.style_down." not in key) and (".adapter.style_up." not in key):
        raise RuntimeError(f"Style state key must be style_down/style_up only, got '{key}'.")

    matched_groups: List[str] = []
    for group_name in STYLE_GROUP_NAMES:
        for adapter_name in group_adapter_names[group_name]:
            if key.startswith(f"{adapter_name}.style_down.") or key.startswith(f"{adapter_name}.style_up."):
                matched_groups.append(group_name)
                break

    if len(matched_groups) != 1:
        raise RuntimeError(f"Style state key '{key}' matched {len(matched_groups)} adapter groups.")
    return matched_groups[0]


def _validate_group_weights(group_name: str, weights: torch.Tensor, expected_len: int) -> torch.Tensor:
    weights_cpu = weights.detach().cpu().to(torch.float32)
    if weights_cpu.shape != (expected_len,):
        raise RuntimeError(
            f"Group '{group_name}' weights must have shape {(expected_len,)}, got {tuple(weights_cpu.shape)}."
        )
    if not bool(torch.isfinite(weights_cpu).all().item()):
        raise RuntimeError(f"Group '{group_name}' weights contain non-finite values: {weights_cpu}")
    if bool((weights_cpu < 0).any().item()):
        raise RuntimeError(f"Group '{group_name}' weights contain negative values: {weights_cpu}")
    weight_sum = float(weights_cpu.sum().item())
    if abs(weight_sum - 1.0) > 1e-4:
        raise RuntimeError(f"Group '{group_name}' weights must sum to 1, got {weight_sum}.")
    return weights_cpu


def _fuse_grouped_style_state(
    client_style_states: Dict[str, Dict[str, torch.Tensor]],
    valid_sources: List[str],
    group_weights: dict[str, torch.Tensor],
    group_adapter_names: dict[str, tuple[str, ...]],
) -> Dict[str, torch.Tensor]:
    for group_name in STYLE_GROUP_NAMES:
        if group_name not in group_weights:
            raise RuntimeError(f"Missing weights for style group '{group_name}'.")
    weights_by_group = {
        group_name: _validate_group_weights(group_name, group_weights[group_name], len(valid_sources))
        for group_name in STYLE_GROUP_NAMES
    }

    if not valid_sources:
        raise RuntimeError("Cannot fuse grouped style state with no valid sources.")
    reference_state = client_style_states[valid_sources[0]]
    style_keys = list(reference_state.keys())
    if not style_keys:
        raise RuntimeError(f"Source '{valid_sources[0]}' has empty style state.")

    for src in valid_sources:
        source_keys = set(client_style_states[src].keys())
        if source_keys != set(style_keys):
            raise RuntimeError(f"Source '{src}' style keys do not match the reference source.")
        for key in source_keys:
            if ("style_down" not in key) and ("style_up" not in key):
                raise RuntimeError(f"Non-style adapter parameter found in source-private style state: '{key}'.")

    fused_style_state: Dict[str, torch.Tensor] = {}
    fused_keys: set[str] = set()
    adapter_style_presence = {
        adapter_name: {"style_down": False, "style_up": False}
        for group_name in STYLE_GROUP_NAMES
        for adapter_name in group_adapter_names[group_name]
    }

    for key in style_keys:
        group_name = _group_for_style_state_key(key, group_adapter_names)
        if key in fused_keys:
            raise RuntimeError(f"Style key '{key}' was fused more than once.")
        fused_keys.add(key)
        for adapter_name in group_adapter_names[group_name]:
            if key.startswith(f"{adapter_name}.style_down."):
                adapter_style_presence[adapter_name]["style_down"] = True
            if key.startswith(f"{adapter_name}.style_up."):
                adapter_style_presence[adapter_name]["style_up"] = True

        weights_cpu = weights_by_group[group_name]
        fused_value: torch.Tensor | None = None
        reference_shape = reference_state[key].shape
        for idx, src in enumerate(valid_sources):
            value = client_style_states[src][key]
            if value.shape != reference_shape:
                raise RuntimeError(
                    f"Shape mismatch for style key '{key}' in source '{src}': "
                    f"expected {tuple(reference_shape)}, got {tuple(value.shape)}."
                )
            weighted = value.detach().cpu().to(torch.float32) * float(weights_cpu[idx].item())
            fused_value = weighted.clone() if fused_value is None else fused_value + weighted
        if fused_value is None:
            raise RuntimeError(f"Failed to fuse style key '{key}'.")
        fused_style_state[key] = fused_value

    missing = [
        f"{adapter_name}.{branch}"
        for adapter_name, presence in adapter_style_presence.items()
        for branch, present in presence.items()
        if not present
    ]
    if missing:
        raise RuntimeError(f"Grouped fused style state is missing required style parameters: {missing}")
    return fused_style_state


def _source_whitened_diag_w2(
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
    reference_scale: torch.Tensor,
) -> torch.Tensor:
    target_mean = target_mean.to(torch.float32)
    target_std = target_std.to(torch.float32)
    source_mean = source_mean.to(torch.float32)
    source_std = source_std.to(torch.float32)
    reference_scale = reference_scale.to(torch.float32)
    mean_term = ((target_mean - source_mean) / reference_scale).square().mean()
    std_term = ((target_std - source_std) / reference_scale).square().mean()
    return torch.sqrt(0.5 * (mean_term + std_term) + 1e-12)


def _build_group_source_w2_geometry(
    client_style_stats: dict[str, dict[str, Any]],
    valid_sources: list[str],
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_name not in STYLE_GROUP_NAMES:
        raise RuntimeError(f"Unknown style group '{group_name}'.")

    group_means = []
    group_stds = []
    expected_shape: torch.Size | None = None
    expected_adapter_names: tuple[str, ...] | None = None
    for src in valid_sources:
        stats = client_style_stats.get(src)
        if not isinstance(stats, dict):
            raise RuntimeError(f"Source '{src}' is missing style stats.")
        all_group_stats = stats.get("group_stats")
        if not isinstance(all_group_stats, dict) or group_name not in all_group_stats:
            raise RuntimeError(f"Source '{src}' is missing grouped raw style stats for group '{group_name}'.")
        group_stats = all_group_stats[group_name]
        if not isinstance(group_stats, dict):
            raise RuntimeError(f"Source '{src}' group '{group_name}' stats must be a dict.")

        count = int(group_stats.get("global_count", 0))
        if count <= 0:
            raise RuntimeError(f"Source '{src}' group '{group_name}' has non-positive global_count={count}.")
        mean = group_stats.get("global_mean")
        std = group_stats.get("global_std")
        if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
            raise RuntimeError(f"Source '{src}' group '{group_name}' mean/std must be tensors.")
        if mean.shape != std.shape:
            raise RuntimeError(
                f"Source '{src}' group '{group_name}' has mismatched mean/std shapes: "
                f"{tuple(mean.shape)} vs {tuple(std.shape)}."
            )
        if mean.numel() == 0:
            raise RuntimeError(f"Source '{src}' group '{group_name}' has empty raw style stats.")
        if expected_shape is None:
            expected_shape = mean.shape
        elif mean.shape != expected_shape:
            raise RuntimeError(
                f"Group '{group_name}' raw style stat shapes must match. "
                f"Expected {tuple(expected_shape)}, got {tuple(mean.shape)} for source '{src}'."
            )
        mean_fp32 = mean.to(torch.float32)
        std_fp32 = std.to(torch.float32)
        if not bool(torch.isfinite(mean_fp32).all().item()):
            raise RuntimeError(f"Source '{src}' group '{group_name}' has non-finite global_mean.")
        if not bool(torch.isfinite(std_fp32).all().item()):
            raise RuntimeError(f"Source '{src}' group '{group_name}' has non-finite global_std.")

        adapter_names_raw = group_stats.get("adapter_names")
        if not isinstance(adapter_names_raw, (tuple, list)):
            raise RuntimeError(f"Source '{src}' group '{group_name}' is missing adapter_names.")
        adapter_names = tuple(str(name) for name in adapter_names_raw)
        if expected_adapter_names is None:
            expected_adapter_names = adapter_names
        elif adapter_names != expected_adapter_names:
            raise RuntimeError(
                f"Group '{group_name}' adapter_names must match across sources. "
                f"Expected {expected_adapter_names}, got {adapter_names} for source '{src}'."
            )

        group_means.append(mean_fp32)
        group_stds.append(std_fp32)

    means = torch.stack(group_means, dim=0)
    stds = torch.stack(group_stds, dim=0)
    center = means.mean(dim=0)
    reference_scale = torch.sqrt(
        stds.square().mean(dim=0)
        + (means - center).square().mean(dim=0)
        + 1e-6
    )
    if not bool(torch.isfinite(reference_scale).all().item()):
        raise RuntimeError(f"Group '{group_name}' reference_scale contains non-finite values.")
    pairwise_distances = []
    for i in range(len(valid_sources)):
        for j in range(i + 1, len(valid_sources)):
            pairwise_distances.append(
                _source_whitened_diag_w2(
                    means[i],
                    stds[i],
                    means[j],
                    stds[j],
                    reference_scale,
                )
            )
    if pairwise_distances:
        source_temperature = torch.stack(pairwise_distances, dim=0).mean().clamp_min(1e-6)
    else:
        source_temperature = torch.tensor(1.0, dtype=torch.float32)
    if not bool(torch.isfinite(source_temperature).all().item()) or float(source_temperature.item()) <= 0.0:
        raise RuntimeError(f"Group '{group_name}' has invalid source_temperature={source_temperature}.")
    return means, stds, reference_scale, source_temperature


def _source_geometry_softmax_weights(
    distances: torch.Tensor,
    source_temperature: torch.Tensor,
) -> torch.Tensor:
    return torch.softmax(
        -distances.to(torch.float32) / source_temperature.to(torch.float32).clamp_min(1e-6),
        dim=0,
    )


def _validate_style_posterior(
    distances: torch.Tensor,
    weights: torch.Tensor,
    source_temperature: torch.Tensor,
) -> None:
    if not bool(torch.isfinite(distances).all().item()):
        raise RuntimeError(f"Invalid raw_batch TTA distances: {distances}")
    if not bool(torch.isfinite(source_temperature).all().item()) or float(source_temperature.item()) <= 0.0:
        raise RuntimeError(f"Invalid raw_batch TTA source_temperature: {source_temperature}")
    if not bool(torch.isfinite(weights).all().item()):
        raise RuntimeError(f"Invalid raw_batch TTA weights: {weights}")
    if bool((weights < 0).any().item()):
        raise RuntimeError(f"Invalid raw_batch TTA negative weights: {weights}")
    weight_sum = float(weights.sum().item())
    if abs(weight_sum - 1.0) > 1e-4:
        raise RuntimeError(f"Invalid raw_batch TTA weight sum: {weight_sum}")


def _valid_style_stat_sources(
    client_style_states: Dict[str, Dict[str, torch.Tensor]],
    client_style_stats: Dict[str, Dict[str, Any]],
    source_domains: List[str],
) -> List[str]:
    valid_sources: List[str] = []
    for src in source_domains:
        style_state = client_style_states.get(src)
        stats = client_style_stats.get(src)
        if not style_state or not isinstance(stats, dict):
            continue
        valid_sources.append(src)
    return valid_sources


def _batch_raw_mean_std(raw_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_fp32 = raw_feats.detach().to(torch.float32).cpu()
    return raw_fp32.mean(dim=0), raw_fp32.std(dim=0, unbiased=False)


def _compute_proto_logits(
    features: torch.Tensor,
    proto_bank: torch.Tensor | None,
    valid_proto_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if proto_bank is None or valid_proto_mask is None:
        return None
    proto_feat = torch.nn.functional.normalize(features.to(torch.float32), dim=-1, eps=1e-6)
    proto_logits = proto_feat @ proto_bank.t()
    return proto_logits.masked_fill(
        ~valid_proto_mask.to(device=proto_logits.device).unsqueeze(0),
        -1e4,
    )


def _ensemble_preds(
    cls_logits: torch.Tensor,
    proto_logits: torch.Tensor | None,
    *,
    prototype_temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_probs = torch.softmax(cls_logits.to(torch.float32), dim=1)
    preds_cls = cls_probs.argmax(dim=1)
    if proto_logits is None:
        return preds_cls, preds_cls, preds_cls
    tau = max(float(prototype_temperature), 1e-6)
    proto_probs = torch.softmax(proto_logits.to(torch.float32) / tau, dim=1)
    ensemble_probs = 0.5 * cls_probs + 0.5 * proto_probs
    return ensemble_probs.argmax(dim=1), preds_cls, proto_probs.argmax(dim=1)


def _raw_batch_distances(
    raw_feats: torch.Tensor,
    source_means: torch.Tensor,
    source_stds: torch.Tensor,
    reference_scale: torch.Tensor,
) -> torch.Tensor:
    target_mean, target_std = _batch_raw_mean_std(raw_feats)
    distances = []
    for idx in range(source_means.size(0)):
        distances.append(
            _source_whitened_diag_w2(
                target_mean,
                target_std,
                source_means[idx],
                source_stds[idx],
                reference_scale,
            )
        )
    return torch.stack(distances, dim=0)


@torch.no_grad()
def evaluate_target_with_style_mode(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device],
    *,
    client_style_states: Dict[str, Dict[str, torch.Tensor]],
    client_style_stats: Dict[str, Dict[str, Any]],
    source_domains: List[str],
    global_prototypes: torch.Tensor | None,
    args: Any,
    logger: logging.Logger,
) -> Tuple[float, float, float, Dict[str, Any]]:
    device = torch.device(device)
    valid_sources = _valid_style_stat_sources(client_style_states, client_style_stats, source_domains)
    if not valid_sources:
        raise RuntimeError("raw_batch TTA requires at least one valid style state.")

    model.eval()
    featurizer = model[0]
    classifier = model[1]
    group_adapter_names = get_style_group_adapter_names(featurizer)
    for src in valid_sources:
        group_stats = client_style_stats[src].get("group_stats")
        if not isinstance(group_stats, dict):
            raise RuntimeError(f"Source '{src}' is missing grouped raw style stats.")
        for group_name in STYLE_GROUP_NAMES:
            stats_adapter_names = tuple(group_stats.get(group_name, {}).get("adapter_names", ()))
            if stats_adapter_names != tuple(group_adapter_names[group_name]):
                raise RuntimeError(
                    f"Source '{src}' group '{group_name}' adapter_names do not match current featurizer. "
                    f"Expected {group_adapter_names[group_name]}, got {stats_adapter_names}."
                )
    group_geometries = {
        group_name: _build_group_source_w2_geometry(client_style_stats, valid_sources, group_name)
        for group_name in STYLE_GROUP_NAMES
    }
    use_amp = device.type == "cuda"
    uniform_weights = torch.full((len(valid_sources),), 1.0 / float(len(valid_sources)), dtype=torch.float32)
    uniform_group_weights = {group_name: uniform_weights for group_name in STYLE_GROUP_NAMES}
    load_adapter_state_to_model(
        model,
        _fuse_grouped_style_state(
            client_style_states,
            valid_sources,
            uniform_group_weights,
            group_adapter_names,
        ),
    )
    proto_bank = None
    valid_proto_mask = None
    if global_prototypes is not None:
        proto_source = global_prototypes.to(device=device, dtype=torch.float32)
        valid_proto_mask = proto_source.norm(dim=-1) > 1e-6
        if bool(valid_proto_mask.any().item()):
            proto_bank = torch.nn.functional.normalize(proto_source, dim=-1, eps=1e-6)
    correct_ensemble = 0
    correct_cls = 0
    correct_proto = 0
    total = 0
    distance_sum = {
        group_name: torch.zeros(len(valid_sources), dtype=torch.float32)
        for group_name in STYLE_GROUP_NAMES
    }
    weight_sum = {
        group_name: torch.zeros(len(valid_sources), dtype=torch.float32)
        for group_name in STYLE_GROUP_NAMES
    }
    batch_count = 0

    try:
        for batch in dataloader:
            imgs = batch[0].to(device, non_blocking=True)
            set_adapter_mode(model, "extract_style")
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = featurizer(imgs)
            raw_feature_groups = collect_style_anchor_feature_groups(featurizer)
            group_weights: dict[str, torch.Tensor] = {}
            for group_name in STYLE_GROUP_NAMES:
                source_means, source_stds, reference_scale, source_temperature = group_geometries[group_name]
                distances = _raw_batch_distances(
                    raw_feature_groups[group_name].detach(),
                    source_means,
                    source_stds,
                    reference_scale,
                )
                weights = _source_geometry_softmax_weights(distances, source_temperature)
                _validate_style_posterior(distances, weights, source_temperature)
                group_weights[group_name] = weights
                distance_sum[group_name] += distances.detach().cpu().to(torch.float32)
                weight_sum[group_name] += weights.detach().cpu().to(torch.float32)

            load_adapter_state_to_model(
                model,
                _fuse_grouped_style_state(
                    client_style_states,
                    valid_sources,
                    group_weights,
                    group_adapter_names,
                ),
            )
            set_adapter_mode(model, "full")
            with torch.amp.autocast("cuda", enabled=use_amp):
                features_full = featurizer(imgs)
                logits = classifier(features_full)
            proto_logits = _compute_proto_logits(features_full, proto_bank, valid_proto_mask)
            preds_ensemble, preds_cls, preds_proto = _ensemble_preds(
                logits,
                proto_logits,
                prototype_temperature=args.prototype_temperature,
            )
            labels = batch[1].to(device, non_blocking=True)
            correct_ensemble += (preds_ensemble == labels).sum().item()
            correct_cls += (preds_cls == labels).sum().item()
            correct_proto += (preds_proto == labels).sum().item()
            total += labels.numel()
            batch_count += 1
    finally:
        set_adapter_mode(model, "full")
    denom_batches = max(1, batch_count)
    group_definition = {
        "early": {
            "adapter_indices": [0, 1, 2, 3],
            "adapter_names": list(group_adapter_names["early"]),
        },
        "middle": {
            "adapter_indices": [4, 5, 6, 7],
            "adapter_names": list(group_adapter_names["middle"]),
        },
        "late": {
            "adapter_indices": [8, 9, 10, 11],
            "adapter_names": list(group_adapter_names["late"]),
        },
    }
    groups_report: Dict[str, Dict[str, Any]] = {}
    for group_name in STYLE_GROUP_NAMES:
        _source_means, _source_stds, reference_scale, source_temperature = group_geometries[group_name]
        groups_report[group_name] = {
            "distances": _tensor_report(valid_sources, distance_sum[group_name] / denom_batches),
            "weights": _tensor_report(valid_sources, weight_sum[group_name] / denom_batches),
            "source_temperature": float(source_temperature.detach().cpu().item()),
            "reference_scale_mean": float(reference_scale.detach().cpu().mean().item()),
            "reference_scale_min": float(reference_scale.detach().cpu().min().item()),
            "reference_scale_max": float(reference_scale.detach().cpu().max().item()),
        }
    report = {
        "mode": "raw_batch_grouped",
        "metric": "grouped_source_whitened_diag_w2",
        "group_definition": group_definition,
        "groups": groups_report,
        "batch_count": int(batch_count),
        "stabilization": "none",
    }
    logger.info("[Test-Time Style] mode=raw_batch_grouped metric=grouped_source_whitened_diag_w2 report=%s", report)
    denom = max(1, total)
    return correct_ensemble / denom, correct_cls / denom, correct_proto / denom, report


@torch.no_grad()
def evaluate_target(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device],
    *,
    global_prototypes: torch.Tensor | None = None,
    prototype_temperature: float = 0.1,
) -> Tuple[float, float, float]:
    model.eval()
    device = torch.device(device)
    correct_ensemble = 0
    correct_cls = 0
    correct_proto = 0
    total = 0
    featurizer = model[0]
    classifier = model[1]
    proto_bank = None
    valid_proto_mask = None
    if global_prototypes is not None:
        proto_source = global_prototypes.to(device=device, dtype=torch.float32)
        valid_proto_mask = proto_source.norm(dim=-1) > 1e-6
        if bool(valid_proto_mask.any().item()):
            proto_bank = torch.nn.functional.normalize(proto_source, dim=-1, eps=1e-6)
    for batch in dataloader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError(f"Unexpected eval batch type: {type(batch)}")
        imgs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        set_adapter_mode(model, "full")
        features_full = featurizer(imgs)
        logits = classifier(features_full)
        proto_scores = _compute_proto_logits(features_full, proto_bank, valid_proto_mask)
        preds_ensemble, preds_cls, preds_proto = _ensemble_preds(
            logits,
            proto_scores,
            prototype_temperature=prototype_temperature,
        )
        correct_ensemble += (preds_ensemble == labels).sum().item()
        correct_cls += (preds_cls == labels).sum().item()
        correct_proto += (preds_proto == labels).sum().item()
        total += labels.numel()
    set_adapter_mode(model, "full")
    denom = max(1.0, float(total))
    return correct_ensemble / denom, correct_cls / denom, correct_proto / denom
