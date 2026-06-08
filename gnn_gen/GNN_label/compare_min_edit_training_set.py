from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Tuple

import torch

from data_prep import DatasetConfig, build_datasets, num_classes_and_offset


# ----------------------------
# Standalone comparison config
# ----------------------------
SEED = 0
PREDICT_BOND_CHANGE = True

# Dataset construction (independent from reaction_dataset_prediction.py).
SELECTED_REACTIONS = None
FILTER_UNIQUE_REACTANTS = False
COPY_COUNTS = [1]
PERTURB_TRAINING = True
PERTURB_SAMPLES = 3
PERTURB_RANGE = 0.1
INCLUDE_BASE_SAMPLE = True
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
INDEX_SCALE = 1.0

# Reaction-level mapping uncertainty:
# compare min-edit and FIFO at each swap setting below.
MAPPING_UNCERTAINTY_SWAPS_OPTIONS = [0, 4]
MAPPING_UNCERTAINTY_SEED = 0


def _base_cfg() -> DatasetConfig:
    return DatasetConfig(
        predict_bond_change=PREDICT_BOND_CHANGE,
        selected_reactions=SELECTED_REACTIONS,
        filter_unique_reactants=FILTER_UNIQUE_REACTANTS,
        perturb_training=PERTURB_TRAINING,
        perturb_samples=PERTURB_SAMPLES,
        perturb_range=PERTURB_RANGE,
        include_base_sample=INCLUDE_BASE_SAMPLE,
        copy_counts=COPY_COUNTS,
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
        show_progress=False,
    )


def _scenario_name(use_min_edit: bool, swaps: int) -> str:
    mode = "min_edit" if use_min_edit else "fifo"
    return f"{mode}_swaps{swaps}"


def _build_cfg(use_min_edit: bool, swaps: int) -> DatasetConfig:
    return replace(
        _base_cfg(),
        use_min_edit_atom_mapping=bool(use_min_edit),
        mapping_uncertainty_swaps=max(0, int(swaps)),
        mapping_uncertainty_seed=int(MAPPING_UNCERTAINTY_SEED),
    )


def _sample_key(d) -> Tuple:
    if hasattr(d, "reaction_ids"):
        rid = tuple(sorted(int(v) for v in d.reaction_ids.view(-1).tolist()))
    elif hasattr(d, "reaction_id"):
        rid = (int(d.reaction_id),)
    else:
        rid = tuple()
    rep = int(getattr(d, "reaction_repeat", 1))
    if hasattr(d, "branch_id_by_group"):
        bid = tuple(int(v) for v in d.branch_id_by_group.view(-1).tolist())
    elif hasattr(d, "branch_id"):
        bid = tuple(int(v) for v in d.branch_id.view(-1).tolist())
    else:
        bid = tuple()
    return rid, rep, bid, int(d.pair_i.numel())


def _class_counts(train_data: List, num_classes: int) -> torch.Tensor:
    counts = torch.zeros((num_classes,), dtype=torch.long)
    for d in train_data:
        y = d.y_edge.view(-1).long().cpu()
        counts += torch.bincount(y, minlength=num_classes)[:num_classes]
    return counts


def _changed_edge_stats(train_data: List, predict_bond_change: bool, delta_offset: int) -> Tuple[float, float]:
    changed = 0
    total = 0
    changed_per_sample: List[float] = []
    for d in train_data:
        y = d.y_edge.view(-1).long().cpu()
        if predict_bond_change:
            mask = (y - delta_offset) != 0
        else:
            react = d.react_edge.view(-1).long().cpu()
            mask = y != react
        c = int(mask.sum().item())
        n = int(mask.numel())
        changed += c
        total += n
        changed_per_sample.append(c)
    frac = float(changed) / float(max(total, 1))
    mean_changed = float(sum(changed_per_sample)) / float(max(len(changed_per_sample), 1))
    return frac, mean_changed


def _compare_labels(train_a: List, train_b: List) -> Dict[str, float]:
    by_key_a = defaultdict(list)
    by_key_b = defaultdict(list)
    for d in train_a:
        by_key_a[_sample_key(d)].append(d.y_edge.view(-1).long().cpu())
    for d in train_b:
        by_key_b[_sample_key(d)].append(d.y_edge.view(-1).long().cpu())

    shared = sorted(set(by_key_a.keys()) & set(by_key_b.keys()))
    only_a = len(set(by_key_a.keys()) - set(by_key_b.keys()))
    only_b = len(set(by_key_b.keys()) - set(by_key_a.keys()))

    exact_samples = 0
    compared_samples = 0
    edge_total = 0
    edge_diff = 0
    l1_total = 0.0

    for k in shared:
        la = by_key_a[k]
        lb = by_key_b[k]
        n = min(len(la), len(lb))
        for i in range(n):
            ya = la[i]
            yb = lb[i]
            if ya.numel() != yb.numel():
                continue
            compared_samples += 1
            if torch.equal(ya, yb):
                exact_samples += 1
            d = (ya != yb)
            edge_diff += int(d.sum().item())
            edge_total += int(ya.numel())
            l1_total += float((ya.float() - yb.float()).abs().mean().item())

    return {
        "shared_keys": float(len(shared)),
        "keys_only_a": float(only_a),
        "keys_only_b": float(only_b),
        "compared_samples": float(compared_samples),
        "exact_sample_match_ratio": float(exact_samples) / float(max(compared_samples, 1)),
        "edge_label_mismatch_ratio": float(edge_diff) / float(max(edge_total, 1)),
        "mean_abs_class_diff_per_sample": float(l1_total) / float(max(compared_samples, 1)),
    }


def _build_bundle(cfg: DatasetConfig):
    random.seed(SEED)
    torch.manual_seed(SEED)
    return build_datasets(cfg)


def _unique_sorted_nonneg(values: List[int]) -> List[int]:
    out = sorted({max(0, int(v)) for v in values})
    return out if out else [0]


def main() -> None:
    num_classes, delta_offset = num_classes_and_offset(PREDICT_BOND_CHANGE)
    swap_options = _unique_sorted_nonneg(MAPPING_UNCERTAINTY_SWAPS_OPTIONS)

    scenarios: Dict[str, Dict] = {}
    for swaps in swap_options:
        for use_min_edit in (True, False):
            name = _scenario_name(use_min_edit, swaps)
            cfg = _build_cfg(use_min_edit=use_min_edit, swaps=swaps)
            bundle = _build_bundle(cfg)
            scenarios[name] = {
                "cfg": cfg,
                "bundle": bundle,
                "class_counts": _class_counts(bundle.train_data, num_classes),
                "changed_stats": _changed_edge_stats(bundle.train_data, PREDICT_BOND_CHANGE, delta_offset),
            }

    print("=== Mapping Comparison: min-edit vs FIFO (standalone) ===")
    print(f"predict_bond_change={PREDICT_BOND_CHANGE} num_classes={num_classes} seed={SEED}")
    print(f"mapping_uncertainty_seed={MAPPING_UNCERTAINTY_SEED} swaps_options={swap_options}")
    print("")

    print("Per-scenario summary:")
    for name in sorted(scenarios.keys()):
        info = scenarios[name]
        bundle = info["bundle"]
        frac, mean_changed = info["changed_stats"]
        print(f"- {name}")
        print(f"  train_samples={len(bundle.train_data)} eval_samples={len(bundle.eval_data)}")
        print(f"  class_counts={info['class_counts'].tolist()}")
        print(f"  changed_edge_fraction={frac:.6f} mean_changed_edges/sample={mean_changed:.4f}")
    print("")

    print("Pairwise: min-edit vs FIFO at fixed swap count")
    for swaps in swap_options:
        a = _scenario_name(True, swaps)
        b = _scenario_name(False, swaps)
        diff = _compare_labels(scenarios[a]["bundle"].train_data, scenarios[b]["bundle"].train_data)
        print(f"- swaps={swaps}: {a} vs {b}")
        print(f"  shared_keys={int(diff['shared_keys'])} compared_samples={int(diff['compared_samples'])}")
        print(f"  exact_sample_match_ratio={diff['exact_sample_match_ratio']:.6f}")
        print(f"  edge_label_mismatch_ratio={diff['edge_label_mismatch_ratio']:.6f}")
        print(f"  mean_abs_class_diff_per_sample={diff['mean_abs_class_diff_per_sample']:.6f}")
    print("")

    if len(swap_options) > 1:
        base_swaps = swap_options[0]
        print(f"Pairwise: swap effect relative to swaps={base_swaps}")
        for use_min_edit in (True, False):
            mode = "min_edit" if use_min_edit else "fifo"
            base_name = _scenario_name(use_min_edit, base_swaps)
            for swaps in swap_options[1:]:
                cmp_name = _scenario_name(use_min_edit, swaps)
                diff = _compare_labels(
                    scenarios[base_name]["bundle"].train_data,
                    scenarios[cmp_name]["bundle"].train_data,
                )
                print(f"- {mode}: {base_name} vs {cmp_name}")
                print(f"  shared_keys={int(diff['shared_keys'])} compared_samples={int(diff['compared_samples'])}")
                print(f"  exact_sample_match_ratio={diff['exact_sample_match_ratio']:.6f}")
                print(f"  edge_label_mismatch_ratio={diff['edge_label_mismatch_ratio']:.6f}")
                print(f"  mean_abs_class_diff_per_sample={diff['mean_abs_class_diff_per_sample']:.6f}")


if __name__ == "__main__":
    main()
