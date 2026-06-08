from __future__ import annotations

import os
import random
from dataclasses import fields
from typing import Any, Dict

import torch

from data_prep import DatasetConfig, build_datasets, num_classes_and_offset
from model import EdgePredictor, ModelConfig
from validate import ValidationConfig, evaluate_and_save


# Checkpoint to load (trained model from reaction_dataset_prediction.py)
MODEL_PATH = "/users/3/du000298/GNN_label/reaction_dataset_prediction/model.pt"

# Inference setup
INFERENCE_COPY_COUNTS = [3,5,7]
SELECTED_REACTIONS = None  # e.g. [1, 6, 13]
FILTER_UNIQUE_REACTANTS = False
INDEX_SCALE = 1.0

# Runtime
DEVICE = "cuda"  # falls back to cpu if cuda is unavailable
SEED = 0

# Inference-only postprocess
INFERENCE_VALENCE_CAP = True
VALENCE_CAPS = {"H": 1, "O": 2}

# Output
OUT_DIR = "reaction_dataset_prediction_inference_copies"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_cfg(raw_cfg: Dict[str, Any]) -> ModelConfig:
    valid = {f.name for f in fields(ModelConfig)}
    kwargs = {k: v for k, v in raw_cfg.items() if k in valid}
    return ModelConfig(**kwargs)


def main() -> None:
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"'state_dict' missing in checkpoint: {MODEL_PATH}")
    if "model_cfg" not in checkpoint:
        raise KeyError(f"'model_cfg' missing in checkpoint: {MODEL_PATH}")

    model_cfg = _load_model_cfg(checkpoint["model_cfg"])
    model = EdgePredictor(model_cfg)
    model.load_state_dict(checkpoint["state_dict"])

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print("[infer] cuda requested but not available; using cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)
    model.to(device)

    predict_bond_change = bool(checkpoint.get("predict_bond_change", True))
    _, delta_offset = num_classes_and_offset(predict_bond_change)

    data_cfg = DatasetConfig(
        predict_bond_change=predict_bond_change,
        selected_reactions=SELECTED_REACTIONS,
        filter_unique_reactants=FILTER_UNIQUE_REACTANTS,
        perturb_training=False,  # inference only
        perturb_samples=0,
        perturb_range=0.0,
        include_base_sample=True,
        copy_counts=INFERENCE_COPY_COUNTS,
        index_scale=INDEX_SCALE,
        show_progress=True,
    )
    bundle = build_datasets(data_cfg)
    if not bundle.eval_data:
        raise RuntimeError("No evaluation samples built. Check selected reactions or copy counts.")

    print(f"[infer] loaded checkpoint: {MODEL_PATH}")
    print(f"[infer] device: {device}")
    print(f"[infer] evaluating {len(bundle.eval_data)} samples with copy_counts={INFERENCE_COPY_COUNTS}")

    val_cfg = ValidationConfig(
        out_dir=OUT_DIR,
        predict_bond_change=predict_bond_change,
        delta_offset=delta_offset,
        apply_valence_cap=INFERENCE_VALENCE_CAP,
        valence_caps=VALENCE_CAPS,
    )

    evaluate_and_save(model=model, bundle=bundle, cfg=val_cfg)
    print(f"[infer] saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
