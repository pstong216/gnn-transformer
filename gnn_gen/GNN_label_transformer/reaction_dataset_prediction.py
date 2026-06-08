from __future__ import annotations

import os
import random

import torch

from data_prep import DatasetConfig, build_datasets, num_classes_and_offset
from model import EdgePredictor, ModelConfig
from plotting import plot_loss
from train import TrainConfig, train_model
from validate import ValidationConfig, evaluate_and_save, evaluate_snapshots_and_save_predicted_graphs


# Core task setup
PREDICT_BOND_CHANGE = True

# Model
HIDDEN = 64
LAYERS = 3
EDGE_HIDDEN = 64
SELF_EPS = 1.0
MOLECULE_BALANCED_POOL = True
USE_BRANCH_FEATURE = True
# Branch mode: "scalar" (simple branch index) or "contextual" (reactant-conditioned learned embedding)
BRANCH_FEATURE_MODE = "contextual"
MAX_BRANCH_SLOTS = 8
BRANCH_EMB_DIM = 8
BRANCH_CONTEXT_DIM = 16
USE_THIRD_BODY_FEATURE = True

# Transformer encoder (inserted after GINEConv layers)
USE_TRANSFORMER = True
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.1

# Edge decoder
DECODER_TYPE = "transformer"  # "mlp" or "transformer"
DECODER_TRANSFORMER_HEADS = 6
DECODER_TRANSFORMER_LAYERS = 2
DECODER_TRANSFORMER_DROPOUT = 0.1
DECODER_MAX_GROUPS = 32

# CVAE-style latent branching (learned branch variable instead of deterministic branch id lookup)
USE_LATENT_BRANCHING = False
LATENT_DIM = 8
LATENT_HIDDEN = 64

# Coupled auxiliary target: predict reaction rate coefficients per reaction-group.
PREDICT_RATE_COEFFICIENTS = True
RATE_OUT_DIM = 6
RATE_HIDDEN = 64
RATE_LOSS_WEIGHT = 1.0

# Optimization
EPOCHS = 200
LR = 2e-4
DEVICE = "cpu"
SEED = 0
LATENT_KL_BETA = 0.05
LATENT_KL_WARMUP_EPOCHS = 100

# Snapshot inference settings
SNAPSHOT_EPOCHS = [1,2,3,4,5, 10,20,30,40,50, 100, 200, 500,1000]
RUN_SNAPSHOT_INFERENCE = False

# Optional single-reaction illustrative demo using the same snapshot checkpoints.
RUN_SINGLE_REACTION_DEMO = False
DEMO_REACTION_ID = 15
DEMO_BRANCH_ID = None  # e.g. 0, 1, ... or None for first match
DEMO_REPEAT_FACTOR = None  # e.g. 1, 2, ... or None for first match
DEMO_DIR_NAME = "demo"
DEMO_NODE_HEATMAP_VMIN = -10.0
DEMO_NODE_HEATMAP_VMAX = 10.0
DEMO_DECODER_HEATMAP_VMIN = -20.0
DEMO_DECODER_HEATMAP_VMAX = 20.0

# Dataset selection
SELECTED_REACTIONS = None  # e.g. [1, 6, 13] reaction ids from hydrogen_adjacency
FILTER_UNIQUE_REACTANTS = False
RANDOM_GROUP_TRAINING = True
RANDOM_GROUP_TRAIN_SAMPLES = 100
RANDOM_GROUP_MIN_SIZE = 1
RANDOM_GROUP_MAX_SIZE = 5
RANDOM_GROUP_SEED = 0
RANDOM_GROUP_TRAINING_REPLACE_BASE = False
RANDOM_GROUP_EVAL = True
RANDOM_GROUP_EVAL_SAMPLES = 10
RANDOM_GROUP_EVAL_MIN_SIZE = 1
RANDOM_GROUP_EVAL_MAX_SIZE = 5
RANDOM_GROUP_EVAL_SEED = 1
RANDOM_GROUP_EVAL_REPLACE_BASE = False
ENSURE_DISJOINT_TRAIN_EVAL = True

# Augmentation
PERTURB_TRAINING = True
PERTURB_SAMPLES = 3
PERTURB_RANGE = 0.1
INDEX_SCALE = 1.0
# Atom mapping toggle:
# True  -> newer minimal-edit mapping (signature + local swap refinement)
# False -> older stable type-FIFO mapping
USE_MIN_EDIT_ATOM_MAPPING = False
# Reaction-level mapping uncertainty:
# apply exactly this many same-type random swaps per reaction mapping.
MAPPING_UNCERTAINTY_SWAPS = 0
# Seed for deterministic reaction-level mapping uncertainty.
MAPPING_UNCERTAINTY_SEED = 0

# Progress
DATA_PREP_PROGRESS = True
TRAIN_PROGRESS = True
TRAIN_LOG_EVERY = 10

# Stabilization
BATCH_SIZE = 4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
USE_LR_SCHEDULER = True
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 40
SCHEDULER_MIN_LR = 1e-6
USE_AMP = False
AMP_DTYPE = "bf16"  # "bf16" or "fp16"

# Inference-only postprocess
INFERENCE_VALENCE_CAP = False
VALENCE_CAPS = {"H": 1, "O": 2}

# Output
OUT_DIR = "reaction_dataset_prediction_transformer_2"
SAVE_MODEL = True
MODEL_NAME = "model.pt"
SAVE_LOSS_HISTORY = True
SAVE_SNAPSHOTS = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    data_cfg = DatasetConfig(
        predict_bond_change=PREDICT_BOND_CHANGE,
        selected_reactions=SELECTED_REACTIONS,
        filter_unique_reactants=FILTER_UNIQUE_REACTANTS,
        perturb_training=PERTURB_TRAINING,
        perturb_samples=PERTURB_SAMPLES,
        perturb_range=PERTURB_RANGE,
        random_group_training=RANDOM_GROUP_TRAINING,
        random_group_train_samples=RANDOM_GROUP_TRAIN_SAMPLES,
        random_group_min_size=RANDOM_GROUP_MIN_SIZE,
        random_group_max_size=RANDOM_GROUP_MAX_SIZE,
        random_group_seed=RANDOM_GROUP_SEED,
        random_group_training_replace_base=RANDOM_GROUP_TRAINING_REPLACE_BASE,
        random_group_eval=RANDOM_GROUP_EVAL,
        random_group_eval_samples=RANDOM_GROUP_EVAL_SAMPLES,
        random_group_eval_min_size=RANDOM_GROUP_EVAL_MIN_SIZE,
        random_group_eval_max_size=RANDOM_GROUP_EVAL_MAX_SIZE,
        random_group_eval_seed=RANDOM_GROUP_EVAL_SEED,
        random_group_eval_replace_base=RANDOM_GROUP_EVAL_REPLACE_BASE,
        ensure_disjoint_train_eval=ENSURE_DISJOINT_TRAIN_EVAL,
        index_scale=INDEX_SCALE,
        use_min_edit_atom_mapping=USE_MIN_EDIT_ATOM_MAPPING,
        mapping_uncertainty_swaps=MAPPING_UNCERTAINTY_SWAPS,
        mapping_uncertainty_seed=MAPPING_UNCERTAINTY_SEED,
        show_progress=DATA_PREP_PROGRESS,
    )

    bundle = build_datasets(data_cfg)
    if not bundle.train_data:
        raise RuntimeError("No training samples were built. Check selection/filters.")

    print(f"Training samples: {len(bundle.train_data)}")
    print(f"Evaluation reactions: {len(bundle.eval_data)}")
    if FILTER_UNIQUE_REACTANTS:
        print("=== Reactions used (unique reactants) ===")
        for eq in bundle.used_equations:
            print(f"  {eq}")
        print("=== Reactions omitted (duplicate reactants) ===")
        for eq in bundle.omitted_equations:
            print(f"  {eq}")

    num_classes, delta_offset = num_classes_and_offset(PREDICT_BOND_CHANGE)

    # If latent branching is enabled, deterministic branch feature is usually redundant.
    use_branch_feature = USE_BRANCH_FEATURE and (not USE_LATENT_BRANCHING)

    model_cfg = ModelConfig(
        node_in=bundle.train_data[0].x.size(1),
        num_classes=num_classes,
        hidden=HIDDEN,
        layers=LAYERS,
        edge_hidden=EDGE_HIDDEN,
        self_eps=SELF_EPS,
        molecule_balanced_pool=MOLECULE_BALANCED_POOL,
        use_branch_feature=use_branch_feature,
        branch_feature_mode=BRANCH_FEATURE_MODE,
        max_branch_slots=MAX_BRANCH_SLOTS,
        branch_emb_dim=BRANCH_EMB_DIM,
        branch_context_dim=BRANCH_CONTEXT_DIM,
        use_third_body_feature=USE_THIRD_BODY_FEATURE,
        use_latent_branching=USE_LATENT_BRANCHING,
        latent_dim=LATENT_DIM,
        latent_hidden=LATENT_HIDDEN,
        predict_rate_coeffs=PREDICT_RATE_COEFFICIENTS,
        rate_out_dim=RATE_OUT_DIM,
        rate_hidden=RATE_HIDDEN,
        use_transformer=USE_TRANSFORMER,
        transformer_heads=TRANSFORMER_HEADS,
        transformer_layers=TRANSFORMER_LAYERS,
        transformer_dropout=TRANSFORMER_DROPOUT,
        decoder_type=DECODER_TYPE,
        decoder_transformer_heads=DECODER_TRANSFORMER_HEADS,
        decoder_transformer_layers=DECODER_TRANSFORMER_LAYERS,
        decoder_transformer_dropout=DECODER_TRANSFORMER_DROPOUT,
        decoder_max_groups=DECODER_MAX_GROUPS,
    )
    model = EdgePredictor(model_cfg)

    train_cfg = TrainConfig(
        epochs=EPOCHS,
        lr=LR,
        device=DEVICE,
        show_progress=TRAIN_PROGRESS,
        log_every=TRAIN_LOG_EVERY,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,
        use_lr_scheduler=USE_LR_SCHEDULER,
        scheduler_factor=SCHEDULER_FACTOR,
        scheduler_patience=SCHEDULER_PATIENCE,
        scheduler_min_lr=SCHEDULER_MIN_LR,
        use_amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        snapshot_epochs=SNAPSHOT_EPOCHS,
        always_snapshot_final=True,
        latent_kl_beta=LATENT_KL_BETA,
        latent_kl_warmup_epochs=LATENT_KL_WARMUP_EPOCHS,
        rate_loss_weight=RATE_LOSS_WEIGHT,
    )
    losses, snapshots = train_model(model=model, train_data=bundle.train_data, num_classes=num_classes, cfg=train_cfg)

    plot_loss(losses, os.path.join(OUT_DIR, "loss.png"))

    if SAVE_LOSS_HISTORY:
        torch.save(losses, os.path.join(OUT_DIR, "losses.pt"))
    if SAVE_SNAPSHOTS:
        torch.save({int(k): v for k, v in snapshots.items()}, os.path.join(OUT_DIR, "snapshots.pt"))

    if SAVE_MODEL:
        checkpoint = {
            "state_dict": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "predict_bond_change": PREDICT_BOND_CHANGE,
            "seed": SEED,
            "rate_mean": bundle.rate_mean,
            "rate_std": bundle.rate_std,
        }
        model_path = os.path.join(OUT_DIR, MODEL_NAME)
        torch.save(checkpoint, model_path)
        print(f"Saved model: {model_path}")

    val_cfg = ValidationConfig(
        out_dir=OUT_DIR,
        predict_bond_change=PREDICT_BOND_CHANGE,
        delta_offset=delta_offset,
        apply_valence_cap=INFERENCE_VALENCE_CAP,
        valence_caps=VALENCE_CAPS,
    )

    if RUN_SNAPSHOT_INFERENCE:
        evaluate_snapshots_and_save_predicted_graphs(
            model=model,
            bundle=bundle,
            cfg=val_cfg,
            snapshots=snapshots,
        )

    evaluate_and_save(model=model, bundle=bundle, cfg=val_cfg)


if __name__ == "__main__":
    main()
