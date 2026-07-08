# Client-side local training.

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.adapters import DecoupledAdapter
from network.get_network import set_adapter_mode

logger = logging.getLogger(__name__)

STYLE_GROUP_NAMES = ("early", "middle", "late")
EXPECTED_ADAPTER_COUNT = 12


def _is_adapter_param_name(name: str) -> bool:
    return "adapter." in name


def _infer_adapter_mode(featurizer: nn.Module) -> bool:
    return any(isinstance(m, DecoupledAdapter) for m in featurizer.modules())


def _clear_adapter_runtime_cache(featurizer: nn.Module) -> None:
    for module in featurizer.modules():
        if isinstance(module, DecoupledAdapter):
            module.last_style_stats_raw = None


def get_style_group_adapter_names(
    featurizer: torch.nn.Module,
) -> dict[str, tuple[str, ...]]:
    """
    Collect DecoupledAdapter names in deterministic featurizer.named_modules() order.

    Current ViT-Base must have exactly 12 adapters.
    early:  index 0-3
    middle: index 4-7
    late:   index 8-11
    """
    adapter_names = [
        name
        for name, module in featurizer.named_modules()
        if isinstance(module, DecoupledAdapter)
    ]
    if len(adapter_names) != EXPECTED_ADAPTER_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_ADAPTER_COUNT} DecoupledAdapter modules, "
            f"found {len(adapter_names)}."
        )

    group_adapter_names: dict[str, tuple[str, ...]] = {
        "early": tuple(adapter_names[0:4]),
        "middle": tuple(adapter_names[4:8]),
        "late": tuple(adapter_names[8:12]),
    }
    assigned_names = []
    for group_name in STYLE_GROUP_NAMES:
        names = group_adapter_names[group_name]
        if len(names) != 4:
            raise RuntimeError(f"Style group '{group_name}' must contain 4 adapters, got {len(names)}.")
        assigned_names.extend(names)
    if len(set(assigned_names)) != EXPECTED_ADAPTER_COUNT:
        raise RuntimeError("Each DecoupledAdapter must belong to exactly one style group.")
    return group_adapter_names


def collect_style_anchor_feature_groups(
    featurizer: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """
    Return grouped raw style anchor features:
    {
        "early":  Tensor[B, D_early],
        "middle": Tensor[B, D_middle],
        "late":   Tensor[B, D_late],
    }
    """
    group_adapter_names = get_style_group_adapter_names(featurizer)
    modules_by_name = dict(featurizer.named_modules())
    feature_groups: dict[str, torch.Tensor] = {}
    expected_batch_size: int | None = None

    for group_name in STYLE_GROUP_NAMES:
        layer_feats = []
        adapter_names = group_adapter_names[group_name]
        if not adapter_names:
            raise RuntimeError(f"Style group '{group_name}' is empty.")
        for adapter_name in adapter_names:
            module = modules_by_name.get(adapter_name)
            if not isinstance(module, DecoupledAdapter):
                raise RuntimeError(f"Style group '{group_name}' contains non-adapter module '{adapter_name}'.")
            style_raw = getattr(module, "last_style_stats_raw", None)
            if not isinstance(style_raw, torch.Tensor):
                raise RuntimeError(f"Missing raw style feature for adapter '{adapter_name}'.")
            if style_raw.dim() != 3 or style_raw.size(1) != 1:
                raise RuntimeError(
                    f"Raw style feature for adapter '{adapter_name}' must have shape [B, 1, D], "
                    f"got {tuple(style_raw.shape)}."
                )
            layer_feat = style_raw.squeeze(1)
            if layer_feat.dim() != 2:
                raise RuntimeError(
                    f"Squeezed raw style feature for adapter '{adapter_name}' must have shape [B, D], "
                    f"got {tuple(layer_feat.shape)}."
                )
            batch_size = int(layer_feat.shape[0])
            if expected_batch_size is None:
                expected_batch_size = batch_size
            elif batch_size != expected_batch_size:
                raise RuntimeError(
                    f"Raw style feature batch size mismatch: expected {expected_batch_size}, "
                    f"got {batch_size} for adapter '{adapter_name}'."
                )
            if not bool(torch.isfinite(layer_feat).all().item()):
                raise RuntimeError(f"Raw style feature for adapter '{adapter_name}' contains non-finite values.")
            layer_feats.append(layer_feat)
        feature_groups[group_name] = torch.cat(layer_feats, dim=1)

    return {group_name: feature_groups[group_name] for group_name in STYLE_GROUP_NAMES}


def _create_loader_order_shadow(train_loader: Iterable):
    # Preserve the original DataLoader shuffle order after progress-log cleanup.
    try:
        return iter(train_loader)
    except TypeError:
        return None


def _close_loader_order_shadow(shadow_iter) -> None:
    if shadow_iter is None:
        return
    shutdown_workers = getattr(shadow_iter, "_shutdown_workers", None)
    if callable(shutdown_workers):
        shutdown_workers()


def setup_trainable_params(model: nn.Module, adapter_mode: bool) -> None:
    featurizer: nn.Module = model[0]
    classifier: nn.Module = model[1]

    if adapter_mode:
        for p in featurizer.parameters():
            p.requires_grad = False

        for name, p in featurizer.named_parameters():
            if _is_adapter_param_name(name):
                p.requires_grad = True
        for p in classifier.parameters():
            p.requires_grad = True
    else:
        for p in featurizer.parameters():
            p.requires_grad = True
        for p in classifier.parameters():
            p.requires_grad = True


def train_client(
    model: nn.Module,
    train_loader: Iterable,
    *,
    device: Union[str, torch.device],
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_name: str = "sgd",
    sgd_momentum: float = 0.9,
    class_loss_label_smoothing: float = 0.0,
    lambda_proto: float = 1.0,
    prototype_state: torch.Tensor | None = None,
    prototype_temperature: float = 0.1,
    log_fn: Callable[[str], None] | None = None,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
    float,
]:
    if log_fn is None:
        log_fn = logger.info
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if lambda_proto < 0.0:
        raise ValueError(f"lambda_proto must be >= 0, got {lambda_proto}")
    if prototype_temperature <= 0.0:
        raise ValueError(f"prototype_temperature must be > 0, got {prototype_temperature}")

    device = torch.device(device)
    model.to(device)
    featurizer: nn.Module = model[0]
    classifier: nn.Module = model[1]
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    adapter_mode = _infer_adapter_mode(featurizer)
    group_adapter_names = get_style_group_adapter_names(featurizer) if adapter_mode else {}
    setup_trainable_params(model, adapter_mode)

    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in trainable_named_params]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable params found for client training.")

    optimizer_name = str(optimizer_name).strip().lower()
    if optimizer_name != "sgd":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. This baseline uses fixed SGD.")
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=lr,
        momentum=float(sgd_momentum),
        weight_decay=weight_decay,
        nesterov=False,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=class_loss_label_smoothing)
    num_classes = int(getattr(classifier, "out_features"))
    feature_dim = int(getattr(classifier, "in_features"))
    if prototype_state is None:
        frozen_protos = torch.zeros((num_classes, feature_dim), device=device, dtype=torch.float32)
    else:
        if prototype_state.shape != (num_classes, feature_dim):
            raise ValueError(
                f"prototype_state shape mismatch, expected {(num_classes, feature_dim)} got {tuple(prototype_state.shape)}"
            )
        frozen_protos = prototype_state.to(device=device, dtype=torch.float32).clone()
    valid_proto_mask = frozen_protos.norm(dim=-1) > 1e-6
    has_valid_prototypes = bool(valid_proto_mask.any().item())
    proto_bank: torch.Tensor | None = None
    invalid_proto_mask: torch.Tensor | None = None
    if has_valid_prototypes:
        proto_bank = F.normalize(frozen_protos, p=2, dim=-1, eps=1e-6)
        invalid_proto_mask = ~valid_proto_mask.unsqueeze(0)

    model.train()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    loss_ce_sum = torch.zeros((), device=device, dtype=torch.float64)
    loss_proto_sum = torch.zeros((), device=device, dtype=torch.float64)
    loss_steps = 0
    proto_skipped_steps = torch.zeros((), device=device, dtype=torch.int64)
    group_raw_style_sum: dict[str, torch.Tensor | None] = {group_name: None for group_name in STYLE_GROUP_NAMES}
    group_raw_style_sq_sum: dict[str, torch.Tensor | None] = {group_name: None for group_name in STYLE_GROUP_NAMES}
    group_raw_style_count: dict[str, int] = {group_name: 0 for group_name in STYLE_GROUP_NAMES}

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_loss_ce_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_loss_proto_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_steps = 0
        order_shadow = _create_loader_order_shadow(train_loader)
        try:
            for batch in train_loader:
                images = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    set_adapter_mode(model, "full")
                    features_full = featurizer(images)
                    logits_full = classifier(features_full)
                    style_feature_groups = collect_style_anchor_feature_groups(featurizer) if adapter_mode else None
                    with torch.no_grad():
                        if style_feature_groups is not None:
                            for group_name in STYLE_GROUP_NAMES:
                                group_fp32 = style_feature_groups[group_name].to(torch.float32)
                                group_batch_sum = group_fp32.sum(dim=0)
                                group_batch_sq_sum = (group_fp32 * group_fp32).sum(dim=0)
                                if group_raw_style_sum[group_name] is None:
                                    group_raw_style_sum[group_name] = torch.zeros_like(group_batch_sum)
                                    group_raw_style_sq_sum[group_name] = torch.zeros_like(group_batch_sq_sum)
                                group_raw_style_sum[group_name] += group_batch_sum
                                group_raw_style_sq_sum[group_name] += group_batch_sq_sum
                                group_raw_style_count[group_name] += int(group_fp32.shape[0])

                    loss_ce = criterion(logits_full, labels)
                    proto_step_skipped = torch.ones((), device=device, dtype=torch.bool)
                    if lambda_proto > 0.0 and has_valid_prototypes:
                        assert proto_bank is not None
                        assert invalid_proto_mask is not None
                        proto_feat = F.normalize(features_full.to(torch.float32), p=2, dim=-1, eps=1e-6)
                        proto_logits = torch.matmul(proto_feat, proto_bank.t()) / prototype_temperature
                        proto_logits = proto_logits.masked_fill(invalid_proto_mask, -1e4)
                        valid_sample_mask = valid_proto_mask[labels]
                        per_sample_proto_loss = F.cross_entropy(proto_logits, labels, reduction="none")
                        valid_weight = valid_sample_mask.to(per_sample_proto_loss.dtype)
                        valid_weight_sum = valid_weight.sum()
                        loss_proto = (per_sample_proto_loss * valid_weight).sum() / valid_weight_sum.clamp_min(1.0)
                        proto_step_skipped = valid_weight_sum <= 0
                    else:
                        loss_proto = loss_ce.new_zeros(())
                    loss = loss_ce + lambda_proto * loss_proto

                scaler.scale(loss).backward()
                with torch.no_grad():
                    loss_sum.add_(loss.detach().to(torch.float64))
                    loss_ce_sum.add_(loss_ce.detach().to(torch.float64))
                    loss_proto_sum.add_(loss_proto.detach().to(torch.float64))
                    epoch_loss_sum.add_(loss.detach().to(torch.float64))
                    epoch_loss_ce_sum.add_(loss_ce.detach().to(torch.float64))
                    epoch_loss_proto_sum.add_(loss_proto.detach().to(torch.float64))
                    proto_skipped_steps.add_(proto_step_skipped.to(torch.int64))
                loss_steps += 1
                epoch_steps += 1
                scaler.step(optimizer)
                scaler.update()
                _clear_adapter_runtime_cache(featurizer)
                del logits_full, loss, loss_ce, loss_proto, features_full
                del images, labels
        finally:
            _close_loader_order_shadow(order_shadow)

        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_avg_total, epoch_avg_ce, epoch_avg_proto = (
            torch.stack(
                [
                    epoch_loss_sum / max(1, epoch_steps),
                    epoch_loss_ce_sum / max(1, epoch_steps),
                    epoch_loss_proto_sum / max(1, epoch_steps),
                ]
            )
            .detach()
            .cpu()
            .tolist()
        )
        proto_skipped_total = int(proto_skipped_steps.detach().cpu().item())
        log_parts = [
            f"[ClientTrain][Epoch {epoch + 1}/{epochs}]",
            f"loss_total={epoch_avg_total:.6f}",
            f"loss_ce={epoch_avg_ce:.6f}",
            f"loss_proto={epoch_avg_proto:.6f}",
        ]
        log_parts.extend(
            [
                f"proto_skipped={proto_skipped_total}",
                f"elapsed={epoch_elapsed:.2f}s",
            ]
        )
        log_fn(" ".join(log_parts))

    model.eval()
    set_adapter_mode(model, "full")
    class_sums = torch.zeros((num_classes, feature_dim), device=device, dtype=torch.float32)
    class_counts = torch.zeros(num_classes, device=device, dtype=torch.float32)

    with torch.inference_mode():
        for batch in train_loader:
            imgs = batch[0].to(device, non_blocking=True)
            lbls = batch[1].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                feats_full = featurizer(imgs)
            proto_feats = F.normalize(feats_full.to(torch.float32), p=2, dim=-1, eps=1e-6)

            label_indices = lbls.to(torch.long)
            class_sums.index_add_(0, label_indices, proto_feats)
            class_counts.add_(
                torch.bincount(label_indices, minlength=num_classes).to(
                    device=device,
                    dtype=torch.float32,
                )
            )
            _clear_adapter_runtime_cache(featurizer)

    new_prototypes = frozen_protos.clone()
    valid_classes = class_counts > 0
    new_prototypes[valid_classes] = class_sums[valid_classes] / class_counts[valid_classes].unsqueeze(1)
    new_prototypes[valid_classes] = F.normalize(
        new_prototypes[valid_classes],
        p=2,
        dim=-1,
        eps=1e-6,
    )

    set_adapter_mode(model, "full")

    adapter_state: Dict[str, torch.Tensor] = {}
    style_state: Dict[str, torch.Tensor] = {}
    if adapter_mode:
        for name, tensor in featurizer.state_dict().items():
            if not _is_adapter_param_name(name):
                continue
            if ("style_down" in name) or ("style_up" in name):
                style_state[name] = tensor.detach().cpu().to(torch.float32)
            else:
                adapter_state[name] = tensor.detach().cpu().to(torch.float32)
    else:
        for name, tensor in featurizer.state_dict().items():
            adapter_state[name] = tensor.detach().cpu().to(torch.float32)

    classifier_state = {
        name: tensor.detach().cpu().to(torch.float32)
        for name, tensor in classifier.state_dict().items()
    }
    prototype_state_out = new_prototypes.detach().cpu().to(torch.float32)
    prototype_class_counts_out = class_counts.detach().cpu().to(torch.float32)

    style_stats: Dict[str, Any] = {}
    group_stats: Dict[str, Dict[str, Any]] = {}
    if adapter_mode:
        expected_group_count: int | None = None
        for group_name in STYLE_GROUP_NAMES:
            group_count = int(group_raw_style_count[group_name])
            group_sum = group_raw_style_sum[group_name]
            group_sq_sum = group_raw_style_sq_sum[group_name]
            if group_sum is None or group_sq_sum is None or group_count <= 0:
                raise RuntimeError(f"Failed to collect training raw style statistics for group '{group_name}'.")
            if expected_group_count is None:
                expected_group_count = group_count
            elif group_count != expected_group_count:
                raise RuntimeError(
                    f"Group '{group_name}' raw style count {group_count} does not match "
                    f"expected_group_count {expected_group_count}."
                )
            group_mean = group_sum / float(group_count)
            group_var = torch.clamp(group_sq_sum / float(group_count) - group_mean * group_mean, min=0.0)
            group_std = torch.sqrt(group_var + 1e-6)
            group_stats[group_name] = {
                "global_count": group_count,
                "global_mean": group_mean.detach().cpu().to(torch.float32),
                "global_std": group_std.detach().cpu().to(torch.float32),
                "adapter_names": tuple(group_adapter_names[group_name]),
            }
    style_stats["group_stats"] = group_stats

    avg_train_loss = float((loss_sum / max(1, loss_steps)).detach().cpu().item())
    return (
        adapter_state,
        style_state,
        classifier_state,
        prototype_state_out,
        prototype_class_counts_out,
        style_stats,
        avg_train_loss,
    )
