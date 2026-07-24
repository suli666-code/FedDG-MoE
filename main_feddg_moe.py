"""FedDG CLI and federated training loop."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import torch
from torch.utils.tensorboard import SummaryWriter

from client_train import train_client
from data.feddg_loaders import (
    build_feddg_dataloaders,
    dataset_domains,
    default_num_classes,
    full_domain_to_code,
    resolve_data_root,
)
from feddg_eval import evaluate_target, evaluate_target_with_style_mode
from feddg_utils import (
    aggregate_main_expert,
    extract_adapter_state_from_model as _extract_adapter_state_from_model,
    extract_classifier_state_from_model as _extract_classifier_state_from_model,
    is_cuda_oom as _is_cuda_oom,
    load_adapter_state_to_model as _load_adapter_state_to_model,
    load_classifier_state_to_model as _load_classifier_state_to_model,
    log_model_param_stats as _log_model_param_stats,
    resolve_num_workers as _resolve_num_workers,
    save_checkpoint as _save_checkpoint,
    seed_everything as _seed_everything,
    setup_run_logger as _setup_run_logger,
)
from network.get_network import GetNetwork, set_adapter_mode, set_router_progress


def _ensemble_weight_tag(cls_weight: float, proto_weight: float) -> str:
    """Return a filesystem-safe identifier for one dual-head fusion setting."""
    return f"cls{cls_weight:.2f}_proto{proto_weight:.2f}".replace(".", "p")


def run_federated_task(
    *,
    args: argparse.Namespace,
    target_domain_full: str,
    expert_order_full: List[str],
    run_stamp: str,
    resolved_num_workers: int,
) -> Dict[str, Any]:
    _seed_everything(args.seed)
    aggregate_style_params = args.style_param_mode == "aggregate"

    domain_to_code = full_domain_to_code if args.dataset == "pacs" else (lambda d: d)
    source_domains_full = [d for d in expert_order_full if d != target_domain_full]
    source_domains_code = [domain_to_code(d) for d in source_domains_full]

    ensemble_weight_tag = _ensemble_weight_tag(args.cls_ensemble_weight, args.proto_ensemble_weight)
    run_name = f"{args.dataset}_{target_domain_full}_{ensemble_weight_tag}_{run_stamp}"
    log_path = os.path.join(args.log_dir, f"train_{run_name}.log")
    tb_run_dir = os.path.join(args.tb_log_dir, run_name)
    logger = _setup_run_logger(
        logger_name=f"feddg_{run_name}",
        log_path=log_path,
    )
    writer = SummaryWriter(log_dir=tb_run_dir)
    ckpt_dir = os.path.join(args.log_dir, "checkpoints")

    global_model = None
    main_expert_state: Dict[str, torch.Tensor] | None = None
    main_classifier_state: Dict[str, torch.Tensor] | None = None
    global_prototype_state: torch.Tensor | None = None
    last_round_payload: Dict[str, Any] | None = None
    best_payload_for_return: Dict[str, Any] | None = None
    current_round_idx: int | None = None
    best_val_acc = float("-inf")
    best_round = -1
    best_val_acc_criterion = (
        "source_val_ensemble_acc_avg"
        f"[cls={args.cls_ensemble_weight:.2f},proto={args.proto_ensemble_weight:.2f}]"
    )

    try:
        logger.info(
            "===== LODO Task Start ===== target=%s sources=%s rounds=%d local_epochs=%d",
            target_domain_full,
            source_domains_full,
            args.comm,
            args.local_epochs,
        )
        logger.info("Task seed reset: seed=%d target=%s", int(args.seed), target_domain_full)
        logger.info("Using device=%s num_workers=%d", args.device, resolved_num_workers)
        logger.info(
            "Dual-head ensemble weights: cls=%.2f proto=%.2f; checkpoint selection uses this source-validation ensemble.",
            args.cls_ensemble_weight,
            args.proto_ensemble_weight,
        )
        logger.info("Args snapshot: %s", json.dumps(vars(args), ensure_ascii=False))
        if aggregate_style_params:
            logger.info(
                "Style parameter mode=aggregate: style_down/style_up join FedAvg; "
                "test-time TTA fusion (%s) is disabled.",
                args.tta_fusion,
            )

        dataloader_dict, _, _, target_domain_code = build_feddg_dataloaders(
            dataset_name=args.dataset,
            data_root=args.data_root,
            target_domain=target_domain_full,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=resolved_num_workers,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )
        class _NetArgs:
            model = "vit_clip"

        global_model, _ = GetNetwork(
            _NetArgs(),
            args.num_classes,
            pretrained=True,
            content_rank=args.content_rank,
            style_rank=args.style_rank,
        )
        global_model.to(args.device)
        _log_model_param_stats(global_model, logger)

        main_expert_state = _extract_adapter_state_from_model(
            global_model,
            include_style_params=aggregate_style_params,
        )
        main_classifier_state = _extract_classifier_state_from_model(global_model)
        classifier_head = global_model[1]
        if not hasattr(classifier_head, "out_features") or not hasattr(classifier_head, "in_features"):
            raise RuntimeError("Classifier must expose in_features/out_features for prototype branch.")
        global_prototype_state = torch.zeros(
            (int(classifier_head.out_features), int(classifier_head.in_features)),
            dtype=torch.float32,
        )

        best_ckpt_path = os.path.join(ckpt_dir, f"best_model_{run_name}.pth")
        best_with_test_ckpt_path = os.path.join(ckpt_dir, f"best_model_{run_name}_with_test.pth")
        target_test_loader = dataloader_dict[target_domain_code]["test"]

        client_style_states: Dict[str, Dict[str, torch.Tensor]] = {}
        client_style_stats: Dict[str, Dict[str, Any]] = {}

        for round_idx in range(args.comm):
            current_round_idx = round_idx
            round_start = time.perf_counter()
            logger.info("----- Round %d/%d -----", round_idx + 1, args.comm)
            router_progress = min(1.0, float(round_idx) / float(max(1, args.router_anneal_rounds)))
            set_router_progress(global_model, router_progress)
            logger.info("Round %d: router_progress=%.4f", round_idx + 1, router_progress)

            client_adapter_states: List[Dict[str, torch.Tensor]] = []
            client_classifier_states: List[Dict[str, torch.Tensor]] = []
            client_prototype_states: List[torch.Tensor] = []
            client_prototype_counts: List[torch.Tensor] = []
            client_sample_counts: List[float] = []
            client_losses: List[float] = []
            proto_ratio = min(1.0, float(round_idx) / float(max(1, args.proto_warmup_rounds)))
            round_lambda_proto = args.lambda_proto * proto_ratio
            logger.info(
                "Round %d: round_lambda_proto=%.6f",
                round_idx + 1,
                round_lambda_proto,
            )

            for client_idx, (src_full, src_code) in enumerate(zip(source_domains_full, source_domains_code)):
                logger.info("[Client %d/%d] domain=%s train start", client_idx + 1, len(source_domains_full), src_full)
                client_model = copy.deepcopy(global_model)
                _load_adapter_state_to_model(client_model, main_expert_state)
                _load_classifier_state_to_model(client_model, main_classifier_state)
                set_router_progress(client_model, router_progress)
                set_adapter_mode(client_model, "full")

                if (
                    not aggregate_style_params
                    and src_code in client_style_states
                    and len(client_style_states[src_code]) > 0
                ):
                    _load_adapter_state_to_model(client_model, client_style_states[src_code])

                (
                    adapter_state,
                    style_state,
                    classifier_state,
                    prototype_state,
                    prototype_class_counts,
                    style_stats,
                    avg_train_loss,
                ) = train_client(
                    model=client_model,
                    train_loader=dataloader_dict[src_code]["train"],
                    device=args.device,
                    epochs=args.local_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    optimizer_name=args.optimizer,
                    sgd_momentum=args.sgd_momentum,
                    lambda_proto=round_lambda_proto,
                    prototype_state=global_prototype_state,
                    prototype_temperature=args.prototype_temperature,
                    aggregate_style_params=aggregate_style_params,
                    log_fn=logger.info,
                )
                if not aggregate_style_params:
                    client_style_states[src_code] = style_state
                    client_style_stats[src_code] = style_stats
                client_adapter_states.append(adapter_state)
                client_classifier_states.append(classifier_state)
                client_prototype_states.append(prototype_state)
                client_prototype_counts.append(prototype_class_counts)
                client_sample_counts.append(float(prototype_class_counts.sum().item()))
                client_losses.append(float(avg_train_loss))

            main_expert_state, main_classifier_state, new_avg_proto = aggregate_main_expert(
                client_adapter_states,
                client_classifier_states,
                client_prototype_states,
                client_sample_counts,
                client_prototype_counts,
            )
            global_prototype_state = new_avg_proto.clone()
            sample_total = sum(client_sample_counts)
            aggregation_weights = {
                src_code: (client_sample_counts[idx] / sample_total if sample_total > 0 else 1.0 / max(1, len(client_sample_counts)))
                for idx, src_code in enumerate(source_domains_code)
            }
            prototype_count_totals = torch.stack(client_prototype_counts, dim=0).sum(dim=0)
            logger.info(
                "Round %d: Aggregation weights=%s prototype_class_count_min=%.1f max=%.1f",
                round_idx + 1,
                aggregation_weights,
                float(prototype_count_totals.min().item()) if prototype_count_totals.numel() > 0 else 0.0,
                float(prototype_count_totals.max().item()) if prototype_count_totals.numel() > 0 else 0.0,
            )
            logger.info("Round %d: Global prototypes updated by sample-count weighted aggregation.", round_idx + 1)
            _load_adapter_state_to_model(global_model, main_expert_state)
            _load_classifier_state_to_model(global_model, main_classifier_state)

            source_val_ensemble_acc_list: List[float] = []
            source_val_cls_acc_list: List[float] = []
            source_val_proto_acc_list: List[float] = []
            source_val_ensemble_acc_by_domain: Dict[str, float] = {}
            source_val_cls_acc_by_domain: Dict[str, float] = {}
            source_val_proto_acc_by_domain: Dict[str, float] = {}
            set_router_progress(global_model, router_progress)
            set_adapter_mode(global_model, "full")
            for src_code in source_domains_code:
                if (
                    not aggregate_style_params
                    and src_code in client_style_states
                    and len(client_style_states[src_code]) > 0
                ):
                    _load_adapter_state_to_model(global_model, client_style_states[src_code])
                src_val_ensemble_acc, src_val_cls_acc, src_val_proto_acc = evaluate_target(
                    global_model,
                    dataloader_dict[src_code]["val"],
                    args.device,
                    global_prototypes=global_prototype_state,
                    prototype_temperature=args.prototype_temperature,
                    cls_ensemble_weight=args.cls_ensemble_weight,
                    proto_ensemble_weight=args.proto_ensemble_weight,
                )
                source_val_ensemble_acc_list.append(float(src_val_ensemble_acc))
                source_val_cls_acc_list.append(float(src_val_cls_acc))
                source_val_proto_acc_list.append(float(src_val_proto_acc))
                source_val_ensemble_acc_by_domain[src_code] = float(src_val_ensemble_acc)
                source_val_cls_acc_by_domain[src_code] = float(src_val_cls_acc)
                source_val_proto_acc_by_domain[src_code] = float(src_val_proto_acc)
            source_val_ensemble_acc_avg = float(sum(source_val_ensemble_acc_list) / max(1, len(source_val_ensemble_acc_list)))
            source_val_cls_acc_avg = float(sum(source_val_cls_acc_list) / max(1, len(source_val_cls_acc_list)))
            source_val_proto_acc_avg = float(sum(source_val_proto_acc_list) / max(1, len(source_val_proto_acc_list)))

            set_adapter_mode(global_model, "full")
            if aggregate_style_params:
                tgt_test_ensemble_acc, tgt_test_cls_acc, tgt_test_proto_acc = evaluate_target(
                    global_model,
                    target_test_loader,
                    args.device,
                    global_prototypes=global_prototype_state,
                    prototype_temperature=args.prototype_temperature,
                    cls_ensemble_weight=args.cls_ensemble_weight,
                    proto_ensemble_weight=args.proto_ensemble_weight,
                )
                tta_report = {
                    "mode": "disabled_global_style_aggregation",
                    "reason": "style_param_mode=aggregate",
                }
            else:
                tgt_test_ensemble_acc, tgt_test_cls_acc, tgt_test_proto_acc, tta_report = evaluate_target_with_style_mode(
                    global_model,
                    target_test_loader,
                    args.device,
                    client_style_states=client_style_states,
                    client_style_stats=client_style_stats,
                    source_domains=source_domains_code,
                    global_prototypes=global_prototype_state,
                    args=args,
                    logger=logger,
                )
            set_adapter_mode(global_model, "full")
            round_train_loss = sum(client_losses) / max(1, len(client_losses))
            round_elapsed = time.perf_counter() - round_start
            selection_score = float(source_val_ensemble_acc_avg)

            tb_step = round_idx + 1
            writer.add_scalar("Source_Val_Ensemble_Accuracy_Avg", float(source_val_ensemble_acc_avg), tb_step)
            writer.add_scalar("Source_Val_Cls_Accuracy_Avg", float(source_val_cls_acc_avg), tb_step)
            writer.add_scalar("Source_Val_Proto_Accuracy_Avg", float(source_val_proto_acc_avg), tb_step)
            writer.add_scalar("Target_Test_Ensemble_Accuracy", float(tgt_test_ensemble_acc), tb_step)
            writer.add_scalar("Target_Test_Cls_Accuracy", float(tgt_test_cls_acc), tb_step)
            writer.add_scalar("Target_Test_Proto_Accuracy", float(tgt_test_proto_acc), tb_step)
            writer.add_scalar("Train_Loss", float(round_train_loss), tb_step)
            writer.add_scalar("Selection_Score", selection_score, tb_step)

            logger.info(
                "[Round %d] source_ensemble_acc_avg=%.4f source_cls_acc_avg=%.4f source_proto_acc_avg=%.4f "
                "target_ensemble_acc=%.4f target_cls_acc=%.4f target_proto_acc=%.4f train_loss=%.6f selection_score=%.4f elapsed=%.1fs",
                round_idx + 1,
                source_val_ensemble_acc_avg,
                source_val_cls_acc_avg,
                source_val_proto_acc_avg,
                float(tgt_test_ensemble_acc),
                float(tgt_test_cls_acc),
                float(tgt_test_proto_acc),
                round_train_loss,
                selection_score,
                round_elapsed,
            )

            last_round_payload = {
                "round": round_idx + 1,
                "target_domain": target_domain_full,
                "ensemble_weights": {
                    "classifier": float(args.cls_ensemble_weight),
                    "prototype": float(args.proto_ensemble_weight),
                },
                "source_val_ensemble_acc_avg": float(source_val_ensemble_acc_avg),
                "source_val_cls_acc_avg": float(source_val_cls_acc_avg),
                "source_val_proto_acc_avg": float(source_val_proto_acc_avg),
                "source_val_ensemble_acc_by_domain": source_val_ensemble_acc_by_domain,
                "source_val_cls_acc_by_domain": source_val_cls_acc_by_domain,
                "source_val_proto_acc_by_domain": source_val_proto_acc_by_domain,
                "best_val_acc_criterion": best_val_acc_criterion,
                "selection_score": selection_score,
                "target_test_ensemble_acc": float(tgt_test_ensemble_acc),
                "target_test_cls_acc": float(tgt_test_cls_acc),
                "target_test_proto_acc": float(tgt_test_proto_acc),
                "router_progress": float(router_progress),
                "main_expert_state": main_expert_state,
                "classifier_state": main_classifier_state,
                "prototype_state": global_prototype_state,
                "client_style_states": client_style_states,
                "client_style_stats": client_style_stats,
                "tta_report": tta_report,
                "run_name": run_name,
                "checkpoint_path": best_ckpt_path,
                "args": vars(args),
            }
            if selection_score >= best_val_acc:
                best_val_acc = selection_score
                best_round = round_idx + 1
                _save_checkpoint(checkpoint_path=best_ckpt_path, payload=last_round_payload, logger=logger)

        if os.path.exists(best_ckpt_path):
            best_payload = torch.load(best_ckpt_path, map_location="cpu")
            if "main_expert_state" in best_payload:
                _load_adapter_state_to_model(global_model, best_payload["main_expert_state"])
            _load_classifier_state_to_model(global_model, best_payload["classifier_state"])
            global_prototype_state = best_payload.get("prototype_state", global_prototype_state)
            best_style_states = best_payload.get("client_style_states", {})
            best_style_stats = best_payload.get("client_style_stats", {})
            best_router_progress = float(
                best_payload.get(
                    "router_progress",
                    min(
                        1.0,
                        float(best_payload["round"] - 1) / float(max(1, args.router_anneal_rounds)),
                    ),
                )
            )
            set_router_progress(global_model, best_router_progress)
            set_adapter_mode(global_model, "full")
            if aggregate_style_params:
                test_ensemble_acc, test_cls_acc, test_proto_acc = evaluate_target(
                    global_model,
                    target_test_loader,
                    args.device,
                    global_prototypes=global_prototype_state,
                    prototype_temperature=args.prototype_temperature,
                    cls_ensemble_weight=args.cls_ensemble_weight,
                    proto_ensemble_weight=args.proto_ensemble_weight,
                )
                final_tta_report = {
                    "mode": "disabled_global_style_aggregation",
                    "reason": "style_param_mode=aggregate",
                }
            else:
                test_ensemble_acc, test_cls_acc, test_proto_acc, final_tta_report = evaluate_target_with_style_mode(
                    global_model,
                    target_test_loader,
                    args.device,
                    client_style_states=best_style_states,
                    client_style_stats=best_style_stats,
                    source_domains=source_domains_code,
                    global_prototypes=global_prototype_state,
                    args=args,
                    logger=logger,
                )
            set_adapter_mode(global_model, "full")
            best_payload["target_test_ensemble_acc"] = float(test_ensemble_acc)
            best_payload["target_test_cls_acc"] = float(test_cls_acc)
            best_payload["target_test_proto_acc"] = float(test_proto_acc)
            best_payload["final_tta_report"] = final_tta_report
            logger.info(
                "[Final] Best round=%d best_selection_score=%.4f target_ensemble_acc=%.4f target_cls_acc=%.4f target_proto_acc=%.4f",
                best_round,
                best_val_acc,
                test_ensemble_acc,
                test_cls_acc,
                test_proto_acc,
            )
            _save_checkpoint(
                checkpoint_path=best_with_test_ckpt_path,
                payload=best_payload,
                logger=logger,
            )
            best_payload_for_return = best_payload
        return best_payload_for_return or last_round_payload or {}
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received; saving an interrupt checkpoint...")
        ckpt_path = os.path.join(ckpt_dir, f"interrupt_{run_name}.pth")
        payload = {
            "round": current_round_idx if current_round_idx is not None else -1,
            "target_domain": target_domain_full,
            "best_val_acc_so_far": best_val_acc,
            "best_val_acc_criterion": best_val_acc_criterion,
            "best_round_so_far": best_round,
            "main_expert_state": main_expert_state,
            "main_classifier_state": main_classifier_state,
            "prototype_state": global_prototype_state,
            "client_style_states": client_style_states,
            "client_style_stats": client_style_stats,
            "last_round_payload": last_round_payload,
            "args": vars(args),
        }
        _save_checkpoint(checkpoint_path=ckpt_path, payload=payload, logger=logger)
        raise
    except RuntimeError as exc:
        if _is_cuda_oom(exc):
            logger.error("CUDA OOM detected; saving an interrupt checkpoint...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ckpt_path = os.path.join(ckpt_dir, f"oom_{run_name}.pth")
            payload = {
                "round": current_round_idx if current_round_idx is not None else -1,
                "target_domain": target_domain_full,
                "main_expert_state": main_expert_state,
                "main_classifier_state": main_classifier_state,
                "prototype_state": global_prototype_state,
                "best_val_acc_criterion": best_val_acc_criterion,
                "client_style_states": client_style_states,
                "client_style_stats": client_style_stats,
                "last_round_payload": last_round_payload,
                "args": vars(args),
                "oom_error": str(exc),
            }
            _save_checkpoint(checkpoint_path=ckpt_path, payload=payload, logger=logger)
        raise
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="FedDG with decoupled adapter training.")

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--dataset", type=str, default="pacs", choices=["pacs", "officehome", "vlcs"])
    data_group.add_argument("--data_root", type=str, default="", help="Dataset root path.")
    data_group.add_argument(
        "--test_domain",
        type=str,
        default="sketch",
        help="Leave-one-domain-out target. Use 'all' to run all targets.",
    )
    data_group.add_argument("--num_workers", type=int, default=2, help="DataLoader workers. -1 means auto.")
    data_group.add_argument("--max_train_samples", type=int, default=0, help="Limit train samples per domain for smoke tests.")
    data_group.add_argument("--max_eval_samples", type=int, default=0, help="Limit val/test samples per domain for smoke tests.")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--num_classes", type=int, default=None)
    train_group.add_argument("--batch_size", type=int, default=32)
    train_group.add_argument("--test_batch_size", type=int, default=64)
    train_group.add_argument("--local_epochs", type=int, default=5)
    train_group.add_argument("--comm", type=int, default=5)

    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument("--lr", type=float, default=1e-3)
    optim_group.add_argument("--sgd_momentum", type=float, default=0.9)
    optim_group.add_argument("--weight_decay", type=float, default=0.0)
    optim_group.add_argument("--lambda_proto", type=float, default=1.0, help="Prototype cosine loss weight.")
    optim_group.add_argument("--proto_tau", dest="prototype_temperature", type=float, default=0.1, help="Prototype cosine CE temperature.")
    optim_group.add_argument("--proto_warmup_rounds", type=int, default=4, help="Rounds for linear prototype-loss warm-up.")
    optim_group.add_argument(
        "--router_anneal_rounds",
        type=int,
        default=None,
        help="Rounds for router temperature cosine annealing. Defaults to --comm.",
    )
    optim_group.add_argument("--content_rank", type=int, default=32, help="Content adapter bottleneck rank.")
    optim_group.add_argument("--style_rank", type=int, default=4, help="Style adapter bottleneck rank.")

    ensemble_group = parser.add_argument_group("Dual-head ensemble")
    ensemble_group.add_argument(
        "--cls_ensemble_weight",
        type=float,
        default=0.5,
        help="Classifier-head probability weight; it must be non-negative and sum to 1 with --proto_ensemble_weight.",
    )
    ensemble_group.add_argument(
        "--proto_ensemble_weight",
        type=float,
        default=0.5,
        help="Prototype-head probability weight; it must be non-negative and sum to 1 with --cls_ensemble_weight.",
    )

    ablation_group = parser.add_argument_group("Ablations")
    ablation_group.add_argument(
        "--tta_fusion",
        choices=["grouped", "global"],
        default="global",
        help="Source-private style TTA fusion: one global W2 weight (default) or per-group W2.",
    )
    ablation_group.add_argument(
        "--style_param_mode",
        choices=["private", "aggregate"],
        default="private",
        help="Keep style parameters source-private (default) or include them in FedAvg and disable TTA.",
    )

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--seed", type=int, default=0)
    runtime_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model compute.",
    )

    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log_dir", type=str, default="training_logs")
    log_group.add_argument("--tb_log_dir", type=str, default="tb_runs")

    args = parser.parse_args()
    args.model = "vit_clip"
    args.optimizer = "sgd"
    if args.prototype_temperature <= 0.0:
        raise ValueError(f"--proto_tau must be > 0, got {args.prototype_temperature}")
    if not math.isfinite(args.cls_ensemble_weight) or not math.isfinite(args.proto_ensemble_weight):
        raise ValueError("--cls_ensemble_weight and --proto_ensemble_weight must be finite.")
    if args.cls_ensemble_weight < 0.0 or args.proto_ensemble_weight < 0.0:
        raise ValueError("--cls_ensemble_weight and --proto_ensemble_weight must be non-negative.")
    ensemble_weight_sum = args.cls_ensemble_weight + args.proto_ensemble_weight
    if abs(ensemble_weight_sum - 1.0) > 1e-6:
        raise ValueError(
            "--cls_ensemble_weight and --proto_ensemble_weight must sum to 1, got "
            f"{ensemble_weight_sum}."
        )
    if args.proto_warmup_rounds < 0:
        raise ValueError(f"--proto_warmup_rounds must be >= 0, got {args.proto_warmup_rounds}")
    if args.router_anneal_rounds is None:
        args.router_anneal_rounds = args.comm
    elif args.router_anneal_rounds <= 0:
        raise ValueError(f"--router_anneal_rounds must be > 0, got {args.router_anneal_rounds}")
    if args.num_classes is None:
        args.num_classes = default_num_classes(args.dataset)
    args.data_root = resolve_data_root(
        args.dataset,
        args.data_root,
        project_dir=os.path.dirname(os.path.abspath(__file__)),
    )

    _seed_everything(args.seed)

    expert_order_full = dataset_domains(args.dataset)
    if args.test_domain != "all" and args.test_domain not in expert_order_full:
        raise ValueError(
            f"For --dataset {args.dataset}, --test_domain must be one of {expert_order_full} or 'all'. "
            f"Got: {args.test_domain}"
        )
    resolved_num_workers = _resolve_num_workers(args.num_workers)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.tb_log_dir, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    targets = expert_order_full if args.test_domain == "all" else [args.test_domain]
    results: List[Dict[str, Any]] = []
    for t in targets:
        results.append(
            run_federated_task(
                args=args,
                target_domain_full=t,
                expert_order_full=expert_order_full,
                run_stamp=run_stamp,
                resolved_num_workers=resolved_num_workers,
            )
        )

    final_summaries = []
    for r in results:
        if "source_val_ensemble_acc_avg" in r:
            final_summaries.append(
                f"{r.get('target_domain')}: source_ensemble_acc_avg={r.get('source_val_ensemble_acc_avg'):.4f}, "
                f"source_cls_acc_avg={r.get('source_val_cls_acc_avg', float('nan')):.4f}, "
                f"source_proto_acc_avg={r.get('source_val_proto_acc_avg', float('nan')):.4f}, "
                f"selection_score={r.get('selection_score', float('nan')):.4f}, "
                f"target_ensemble_acc={r.get('target_test_ensemble_acc', float('nan')):.4f}, "
                f"target_cls_acc={r.get('target_test_cls_acc', float('nan')):.4f}, "
                f"target_proto_acc={r.get('target_test_proto_acc', float('nan')):.4f}"
            )
    if final_summaries:
        print("[Summary] " + "; ".join(final_summaries))


if __name__ == "__main__":
    main()
