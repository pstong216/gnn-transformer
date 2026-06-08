from __future__ import annotations

import os
import random
from dataclasses import fields
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import matplotlib as mpl
import torch

from demo_monitor import DemoMonitorConfig, run_single_reaction_demo
from model import EdgePredictor, ModelConfig


# Paths
OUT_DIR = "/users/3/du000298/GNN_label/reaction_dataset_prediction_rate1"
MODEL_PATH = os.path.join(OUT_DIR, "model.pt")
DEMO_DIR_NAME = "demo"

# Demo target
DEMO_REACTION_ID = 15
DEMO_BRANCH_ID = None
DEMO_REPEAT_FACTOR = 1
INDEX_SCALE = 1.0
NODE_HEATMAP_VMIN = -5
NODE_HEATMAP_VMAX = 15.0
DECODER_HEATMAP_VMIN = -5
DECODER_HEATMAP_VMAX = 15

# Epoch selection / refresh behavior
SNAPSHOT_PATH = os.path.join(OUT_DIR, "snapshots.pt")  # optional future artifact
LOSS_PATH = os.path.join(OUT_DIR, "losses.pt")  # optional future artifact
USE_EXISTING_EPOCH_DIRS = True
FALLBACK_EPOCHS = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200]

# Runtime
DEVICE = "cuda"  # falls back to cpu
SEED = 0

# Figure font sizes (global style for all demo plots)
FIG_FONT_SIZE = 14
FIG_TITLE_SIZE = 16
FIG_TICK_SIZE = 14
FIG_LEGEND_SIZE = 14


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": FIG_FONT_SIZE,
            "axes.titlesize": FIG_TITLE_SIZE,
            "axes.labelsize": FIG_FONT_SIZE,
            "xtick.labelsize": FIG_TICK_SIZE,
            "ytick.labelsize": FIG_TICK_SIZE,
            "legend.fontsize": FIG_LEGEND_SIZE,
            "figure.titlesize": FIG_TITLE_SIZE,
        }
    )


def _load_model_cfg(raw_cfg: Dict[str, Any]) -> ModelConfig:
    valid = {f.name for f in fields(ModelConfig)}
    kwargs = {k: v for k, v in raw_cfg.items() if k in valid}
    return ModelConfig(**kwargs)


def _parse_epoch_dirs(base_demo_dir: str) -> List[int]:
    if not os.path.isdir(base_demo_dir):
        return []
    out: List[int] = []
    for name in os.listdir(base_demo_dir):
        if not name.startswith("epoch_"):
            continue
        try:
            out.append(int(name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(out))


def _find_existing_reaction_demo_dir(
    demo_root: str,
    reaction_id: int,
    branch_id: Optional[int],
    repeat_factor: int,
) -> Optional[str]:
    rid = f"reaction_{int(reaction_id):02d}_"
    rep = f"_x{int(repeat_factor)}"
    if not os.path.isdir(demo_root):
        return None

    candidates: List[str] = []
    for name in sorted(os.listdir(demo_root)):
        full = os.path.join(demo_root, name)
        if not os.path.isdir(full):
            continue
        if not name.startswith(rid):
            continue
        if rep not in name:
            continue
        candidates.append(full)

    if not candidates:
        return None
    if branch_id is None:
        return candidates[0]

    exact = os.path.join(demo_root, f"reaction_{int(reaction_id):02d}_b{int(branch_id)}_x{int(repeat_factor)}")
    if os.path.isdir(exact):
        return exact
    return candidates[0]


def _load_optional_losses(checkpoint: Dict[str, Any]) -> Optional[List[float]]:
    losses = checkpoint.get("losses")
    if isinstance(losses, list) and losses:
        return [float(x) for x in losses]
    if os.path.exists(LOSS_PATH):
        obj = torch.load(LOSS_PATH, map_location="cpu")
        if isinstance(obj, list) and obj:
            return [float(x) for x in obj]
    return None


def _build_snapshots(
    final_state: Dict[str, torch.Tensor],
    epoch_list: List[int],
) -> Dict[int, Dict[str, torch.Tensor]]:
    return {int(e): final_state for e in epoch_list}


def main() -> None:
    _set_seed(SEED)
    _set_figure_style()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    if "state_dict" not in ckpt or "model_cfg" not in ckpt:
        raise KeyError(f"Checkpoint missing required keys ('state_dict'/'model_cfg'): {MODEL_PATH}")

    model_cfg = _load_model_cfg(ckpt["model_cfg"])
    model = EdgePredictor(model_cfg)
    model.load_state_dict(ckpt["state_dict"])

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print("[refresh_demo] cuda requested but not available; using cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)
    model.to(device)

    losses = _load_optional_losses(ckpt)
    if losses is None:
        print("[refresh_demo] warning: no loss history found; convergence plot will show 'unavailable'.")

    demo_cfg = DemoMonitorConfig(
        out_dir=os.path.join(OUT_DIR, DEMO_DIR_NAME),
        reaction_id=DEMO_REACTION_ID,
        branch_id=DEMO_BRANCH_ID,
        repeat_factor=DEMO_REPEAT_FACTOR,
        predict_bond_change=bool(ckpt.get("predict_bond_change", True)),
        delta_offset=3,
        index_scale=INDEX_SCALE,
        embedding_grid=(8, 8),
        node_embedding_vmin=NODE_HEATMAP_VMIN,
        node_embedding_vmax=NODE_HEATMAP_VMAX,
        decoder_vmin=DECODER_HEATMAP_VMIN,
        decoder_vmax=DECODER_HEATMAP_VMAX,
        dpi=220,
    )

    # Decide epoch list.
    sample_dir = _find_existing_reaction_demo_dir(
        demo_root=demo_cfg.out_dir,
        reaction_id=DEMO_REACTION_ID,
        branch_id=DEMO_BRANCH_ID,
        repeat_factor=DEMO_REPEAT_FACTOR,
    ) or os.path.join(
        demo_cfg.out_dir,
        f"reaction_{DEMO_REACTION_ID:02d}_b{DEMO_BRANCH_ID or 0}_x{DEMO_REPEAT_FACTOR}",
    )
    epoch_list = _parse_epoch_dirs(sample_dir) if USE_EXISTING_EPOCH_DIRS else []
    if not epoch_list:
        epoch_list = list(FALLBACK_EPOCHS)

    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}

    # Prefer true snapshot states if available.
    if os.path.exists(SNAPSHOT_PATH):
        snap_obj = torch.load(SNAPSHOT_PATH, map_location="cpu")
        if isinstance(snap_obj, dict) and snap_obj:
            use_epochs = sorted(set(int(e) for e in epoch_list if int(e) in snap_obj))
            if use_epochs:
                snapshots = {int(e): snap_obj[int(e)] for e in use_epochs}
                print(f"[refresh_demo] using snapshot states from: {SNAPSHOT_PATH}")

    # Fallback: reuse final model state across requested epochs.
    if not snapshots:
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        snapshots = _build_snapshots(final_state, epoch_list)
        print("[refresh_demo] warning: no snapshot states found; using final model state for all epochs.")

    # Minimal bundle-like object needed by demo_monitor (for rate de-standardization).
    rate_mean = ckpt.get("rate_mean", torch.zeros(6, dtype=torch.float32))
    rate_std = ckpt.get("rate_std", torch.ones(6, dtype=torch.float32))
    bundle_proxy = SimpleNamespace(rate_mean=rate_mean, rate_std=rate_std)

    run_single_reaction_demo(
        model=model,
        bundle=bundle_proxy,
        snapshots=snapshots,
        cfg=demo_cfg,
        losses=losses,
    )
    print(f"[refresh_demo] updated demo at: {demo_cfg.out_dir}")


if __name__ == "__main__":
    main()
