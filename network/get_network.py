# Model construction utilities.

from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
import timm

from network.adapters import DecoupledAdapter


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    # Set requires_grad for all parameters under module.
    for p in module.parameters():
        p.requires_grad = requires_grad


def _find_modules(
    model: nn.Module,
    ancestor_class: Optional[Set[str]] = None,
    search_class: Optional[Iterable[Type[nn.Module]]] = None,
    exclude_children_of: Optional[List[Type[nn.Module]]] = None,
):
    # Find modules by class and return parent, child_name, child, full_name.
    if search_class is None:
        search_class = (nn.Linear,)
    else:
        search_class = tuple(search_class)

    if ancestor_class is None:
        for parent_fullname, parent in model.named_modules():
            for name, module in parent.named_children():
                if any(isinstance(module, _class) for _class in search_class):
                    if exclude_children_of and any(isinstance(parent, _class) for _class in exclude_children_of):
                        continue
                    full_name = f"{parent_fullname}.{name}" if parent_fullname else name
                    yield parent, name, module, full_name
        return

    ancestors = (module for module in model.modules() if module.__class__.__name__ in ancestor_class)
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if fullname == "":
                continue
            if any(isinstance(module, _class) for _class in search_class):
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                if exclude_children_of and any(isinstance(parent, _class) for _class in exclude_children_of):
                    continue
                yield parent, name, module, fullname


class BypassWrapper(nn.Module):
    # Bypass wrapper: output = base_linear(x) + adapter(x).

    def __init__(self, linear_layer: nn.Module, adapter: nn.Module):
        super().__init__()
        self.linear = linear_layer
        self.adapter = adapter
        self.adapter_mode: str = "full"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        return base + self.adapter(x, target_inference_mode=self.adapter_mode)


def inject_trainable_invariant_adapter(
    model: nn.Module,
    r: int = 16,
    style_r: int = 4,
    target_replace_module: Optional[Set[str]] = None,
) -> None:
    # Inject DecoupledAdapter in ViT MLP output layers.
    targets: List[Tuple[nn.Module, str, nn.Module, str]] = list(
        _find_modules(
            model,
            ancestor_class=target_replace_module,
            search_class=[nn.Linear],
        )
    )
    valid_target_names = ["fc2", "c_proj", "mlp.fc2", "mlp.c_proj"]

    for _module, name, _child_module, fullname in targets:
        if not any(name.endswith(target) or fullname.endswith(target) for target in valid_target_names):
            continue

        already = _module._modules.get(name, None)
        if isinstance(already, BypassWrapper) and isinstance(getattr(already, "adapter", None), DecoupledAdapter):
            continue

        base_layer = _child_module
        if not isinstance(base_layer, nn.Linear):
            continue

        adapter = DecoupledAdapter(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            content_r=r,
            style_r=style_r,
        )
        wrapper = BypassWrapper(base_layer, adapter)
        wrapper.to(base_layer.weight.device).to(base_layer.weight.dtype)
        _module._modules[name] = wrapper


def _replace_vit_head_with_identity(vit_model: nn.Module) -> None:
    # Replace ViT head with Identity if present.
    if hasattr(vit_model, "head"):
        setattr(vit_model, "head", nn.Identity())


def set_adapter_mode(model: nn.Module, mode: str) -> None:
    # Set adapter mode for all wrappers: full/extract_style.
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"full", "extract_style"}:
        raise ValueError(f"Unknown adapter mode: {mode}")
    for module in model.modules():
        if isinstance(module, BypassWrapper):
            module.adapter_mode = mode_norm


def set_router_progress(model: nn.Module, progress: float) -> None:
    # Set router annealing progress for all decoupled adapters.
    progress_clamped = min(1.0, max(0.0, float(progress)))
    for module in model.modules():
        if isinstance(module, DecoupledAdapter):
            module.set_router_progress(progress_clamped)


def _local_hf_cache_dir() -> str:
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(project_dir), "hf_hub_cache")


def _load_local_timm_hf_weights(model: nn.Module, hf_hub_id: str) -> str:
    cache_dir = _local_hf_cache_dir()
    repo_cache = os.path.join(cache_dir, "models--" + hf_hub_id.replace("/", "--"))
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    if not os.path.isdir(snapshots_dir):
        raise FileNotFoundError(f"Local HF cache snapshot dir not found: {snapshots_dir}")

    snapshot_dirs = [
        os.path.join(snapshots_dir, name)
        for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    ]
    if not snapshot_dirs:
        raise FileNotFoundError(f"No local HF snapshots found in: {snapshots_dir}")
    snapshot_dirs.sort(key=lambda path: os.path.getmtime(path), reverse=True)

    last_error: Exception | None = None
    for snapshot_dir in snapshot_dirs:
        safetensors_path = os.path.join(snapshot_dir, "model.safetensors")
        torch_path = os.path.join(snapshot_dir, "pytorch_model.bin")
        if os.path.isfile(safetensors_path):
            try:
                from safetensors.torch import load_file

                state_dict = load_file(safetensors_path, device="cpu")
                model.load_state_dict(state_dict, strict=True)
                return safetensors_path
            except Exception as exc:
                last_error = exc
        if os.path.isfile(torch_path):
            try:
                state_dict = torch.load(torch_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=True)
                return torch_path
            except Exception as exc:
                last_error = exc

    if last_error is not None:
        raise RuntimeError(f"Failed to load local HF weights from {snapshots_dir}: {last_error}") from last_error
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in: {snapshots_dir}")


def GetNetwork(
    args: Any,
    num_classes: int,
    pretrained: bool = True,
    **kwargs: Any,
) -> Tuple[nn.Module, int]:
    # Build and return nn.Sequential(featurizer, classifier).
    model_name = getattr(args, "model", None)
    if model_name is None:
        raise ValueError("args.model is required.")

    content_rank = kwargs.get("content_rank", getattr(args, "content_rank", 16))
    style_rank = kwargs.get("style_rank", getattr(args, "style_rank", 4))
    if model_name in ["vit_base_patch16_224", "vit_clip"]:
        timm_name = (
            "vit_base_patch16_224"
            if model_name == "vit_base_patch16_224"
            else "vit_base_patch16_clip_224.laion2b"
        )
        if pretrained and model_name == "vit_clip":
            featurizer = timm.create_model(timm_name, pretrained=False)
            _load_local_timm_hf_weights(featurizer, "timm/vit_base_patch16_clip_224.laion2b")
        else:
            featurizer = timm.create_model(timm_name, pretrained=pretrained)
        feature_level = featurizer.num_features
        _replace_vit_head_with_identity(featurizer)
        _set_requires_grad(featurizer, requires_grad=False)
        inject_trainable_invariant_adapter(
            featurizer,
            r=content_rank,
            style_r=style_rank,
        )
        classifier = nn.Linear(in_features=feature_level, out_features=num_classes, bias=True)
        return nn.Sequential(featurizer, classifier), feature_level

    raise ValueError(f"Unsupported args.model='{model_name}'.")
