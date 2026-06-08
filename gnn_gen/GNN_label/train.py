from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import time

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

from model import EdgePredictor


@dataclass
class TrainConfig:
    epochs: int = 500
    lr: float = 1e-4
    device: str = "cpu"
    show_progress: bool = True
    log_every: int = 10
    batch_size: int = 1
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0

    use_lr_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 40
    scheduler_min_lr: float = 1e-6

    snapshot_epochs: Optional[List[int]] = None
    always_snapshot_final: bool = True

    # CVAE latent-branch regularization: total loss = CE + beta * KL.
    latent_kl_beta: float = 0.05
    latent_kl_warmup_epochs: int = 100

    # Auxiliary regression loss for rate coefficients:
    # total loss = CE + beta*KL + rate_loss_weight*rate_loss.
    rate_loss_weight: float = 1.0

    # Automatic mixed precision (CUDA only).
    use_amp: bool = False
    amp_dtype: str = "bf16"  # "bf16" or "fp16"


def compute_class_weights(train_data, num_classes: int) -> torch.Tensor:
    all_edges = torch.cat([d.y_edge for d in train_data], dim=0)
    counts = torch.bincount(all_edges, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * (counts + 1e-6))
    return weights / weights.mean()


def _snapshot_state_dict(model: EdgePredictor) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _rate_regression_loss(aux: Dict, sample: Data, device: torch.device) -> torch.Tensor:
    pred = aux.get("rate_pred_by_group")
    if pred is None:
        return torch.zeros((), dtype=torch.float32, device=device)
    if not hasattr(sample, "rate_target_by_group"):
        return torch.zeros((), dtype=torch.float32, device=device)

    target = sample.rate_target_by_group.float().to(device)
    if target.dim() == 1:
        target = target.view(1, -1)

    if hasattr(sample, "rate_mask_by_group"):
        mask = sample.rate_mask_by_group.float().to(device)
        if mask.dim() == 1:
            mask = mask.view(1, -1)
    else:
        mask = torch.ones_like(target)

    n = min(pred.size(0), target.size(0))
    d = min(pred.size(1), target.size(1))
    pred = pred[:n, :d]
    target = target[:n, :d]
    mask = mask[:n, :d]

    denom = torch.clamp(mask.sum(), min=1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def _build_train_loader(train_data, batch_size: int):
    if batch_size <= 1:
        return GeoDataLoader(train_data, batch_size=1, shuffle=True), "geo"

    # Variable-sized graph-level tensors (e.g., react_adj of shape NxN) cannot be
    # collated by PyG Batch for batch_size>1. Use list-collation and accumulate.
    return (
        TorchDataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,
        ),
        "list",
    )


def _step_single_batch(
    model: EdgePredictor,
    batch: Data,
    class_weights: torch.Tensor,
    device: torch.device,
    latent_beta: float,
    rate_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, float, float, float]:
    batch = batch.to(device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
    with amp_ctx:
        out = model(batch, return_aux=True)
        if isinstance(out, tuple):
            logits, aux = out
        else:
            logits, aux = out, {}

        ce_loss = F.cross_entropy(logits, batch.y_edge.view(-1), weight=class_weights)
        kl_loss = aux.get("kl_loss")
        if kl_loss is None:
            kl_loss = torch.zeros((), dtype=torch.float32, device=device)
        rate_loss = _rate_regression_loss(aux, batch, device=device)
        loss = ce_loss + latent_beta * kl_loss + rate_weight * rate_loss
    return loss, float(ce_loss.item()), float(kl_loss.item()), float(rate_loss.item())


def _step_list_batch(
    model: EdgePredictor,
    batch_list: List[Data],
    class_weights: torch.Tensor,
    device: torch.device,
    latent_beta: float,
    rate_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, float, float, float]:
    # Compute mean loss over the sample list to emulate minibatch training
    # without requiring fixed-size tensor collation.
    total = torch.tensor(0.0, device=device)
    ce_total = 0.0
    kl_total = 0.0
    rate_total = 0.0
    for sample in batch_list:
        sample = sample.to(device)
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
        with amp_ctx:
            out = model(sample, return_aux=True)
            if isinstance(out, tuple):
                logits, aux = out
            else:
                logits, aux = out, {}

            ce_loss = F.cross_entropy(logits, sample.y_edge.view(-1), weight=class_weights)
            kl_loss = aux.get("kl_loss")
            if kl_loss is None:
                kl_loss = torch.zeros((), dtype=torch.float32, device=device)
            rate_loss = _rate_regression_loss(aux, sample, device=device)
            total = total + (ce_loss + latent_beta * kl_loss + rate_weight * rate_loss)
        ce_total += float(ce_loss.item())
        kl_total += float(kl_loss.item())
        rate_total += float(rate_loss.item())

    denom = max(1, len(batch_list))
    return total / denom, ce_total / denom, kl_total / denom, rate_total / denom


def train_model(
    model: EdgePredictor,
    train_data,
    num_classes: int,
    cfg: TrainConfig,
) -> Tuple[List[float], Dict[int, Dict[str, torch.Tensor]]]:
    device = torch.device(cfg.device)
    model.to(device)

    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    amp_dtype_name = str(cfg.amp_dtype).strip().lower()
    if amp_dtype_name not in {"bf16", "bfloat16", "fp16", "float16"}:
        raise ValueError("amp_dtype must be one of: 'bf16', 'bfloat16', 'fp16', 'float16'")
    if amp_enabled and amp_dtype_name in {"bf16", "bfloat16"}:
        bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        amp_dtype = torch.float16
    scaler_enabled = bool(amp_enabled and amp_dtype == torch.float16)
    scaler: Optional[Union[torch.cuda.amp.GradScaler, torch.amp.GradScaler]]
    if scaler_enabled:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    class_weights = compute_class_weights(train_data, num_classes).to(device)
    loader, loader_mode = _build_train_loader(train_data, batch_size=max(1, cfg.batch_size))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = None
    if cfg.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )

    snapshot_targets = set()
    if cfg.snapshot_epochs:
        snapshot_targets.update(int(e) for e in cfg.snapshot_epochs if 1 <= int(e) <= cfg.epochs)
    if cfg.always_snapshot_final:
        snapshot_targets.add(cfg.epochs)

    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
    losses: List[float] = []
    start_time = time.time()

    if cfg.show_progress:
        print(
            f"[train] start: epochs={cfg.epochs}, batches/epoch={len(loader)}, "
            f"batch_size={max(1, cfg.batch_size)}, mode={loader_mode}, device={device.type}, "
            f"amp={amp_enabled}, amp_dtype={str(amp_dtype).replace('torch.', '')}, scaler={scaler_enabled}"
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        running_ce = 0.0
        running_kl = 0.0
        running_rate = 0.0
        if cfg.latent_kl_warmup_epochs > 0:
            warm = min(1.0, float(epoch) / float(cfg.latent_kl_warmup_epochs))
        else:
            warm = 1.0
        latent_beta = cfg.latent_kl_beta * warm

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            if loader_mode == "geo":
                loss, ce_val, kl_val, rate_val = _step_single_batch(
                    model,
                    batch,
                    class_weights,
                    device,
                    latent_beta=latent_beta,
                    rate_weight=cfg.rate_loss_weight,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            else:
                loss, ce_val, kl_val, rate_val = _step_list_batch(
                    model,
                    batch,
                    class_weights,
                    device,
                    latent_beta=latent_beta,
                    rate_weight=cfg.rate_loss_weight,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                optimizer.step()
            running += float(loss.item())
            running_ce += float(ce_val)
            running_kl += float(kl_val)
            running_rate += float(rate_val)

        losses.append(running)
        avg_loss = running / max(1, len(loader))

        if scheduler is not None:
            scheduler.step(avg_loss)

        if epoch in snapshot_targets:
            snapshots[epoch] = _snapshot_state_dict(model)
            if cfg.show_progress:
                print(f"[train] snapshot saved at epoch {epoch}")

        if cfg.show_progress:
            should_log = epoch == 1 or epoch == cfg.epochs or (cfg.log_every > 0 and epoch % cfg.log_every == 0)
            if should_log:
                elapsed = time.time() - start_time
                ep_per_sec = epoch / max(elapsed, 1e-9)
                eta = (cfg.epochs - epoch) / max(ep_per_sec, 1e-9)
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[train] epoch {epoch:4d}/{cfg.epochs} "
                    f"loss={running:.6f} avg={avg_loss:.6f} "
                    f"ce={running_ce/max(1,len(loader)):.6f} "
                    f"kl={running_kl/max(1,len(loader)):.6f} "
                    f"rate={running_rate/max(1,len(loader)):.6f} "
                    f"rate_w={cfg.rate_loss_weight:.3f} "
                    f"beta={latent_beta:.4f} lr={lr_now:.2e} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
                )

    if cfg.show_progress:
        total_elapsed = time.time() - start_time
        print(f"[train] done in {total_elapsed:.1f}s")

    return losses, snapshots
