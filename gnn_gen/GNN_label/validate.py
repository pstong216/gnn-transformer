from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os

import torch

from data_prep import DatasetBundle, decode_rate_triplet
from hydrogen_adjacency import REACTIONS
from model import EdgePredictor
from plotting import plot_graph


@dataclass
class ValidationConfig:
    out_dir: str
    predict_bond_change: bool = True
    delta_offset: int = 3
    apply_valence_cap: bool = False
    valence_caps: Optional[Dict[str, int]] = None  # e.g. {"H": 1, "O": 2}
    save_rate_predictions: bool = True


def _reconstruct_pred_matrix(data, pred_edges: torch.Tensor, cfg: ValidationConfig) -> torch.Tensor:
    n = int(data.n_nodes)
    pred_mat = torch.zeros((n, n), dtype=torch.float32)
    for k, (i, j) in enumerate(zip(data.pair_i.tolist(), data.pair_j.tolist())):
        if cfg.predict_bond_change:
            delta = float(pred_edges[k].item() - cfg.delta_offset)
            pred_mat[i, j] = data.react_adj[i, j] + delta
            pred_mat[j, i] = data.react_adj[j, i] + delta
        else:
            bond = float(pred_edges[k].item())
            pred_mat[i, j] = bond
            pred_mat[j, i] = bond
    return pred_mat.clamp(0, 3)


def _apply_valence_cap(
    pred_mat: torch.Tensor,
    atom_types: List[str],
    pair_i: List[int],
    pair_j: List[int],
    edge_conf: List[float],
    valence_caps: Optional[Dict[str, int]],
) -> torch.Tensor:
    if not valence_caps:
        return pred_mat

    capped = pred_mat.clone()

    def cap(atom: str) -> float:
        if atom in valence_caps:
            return float(valence_caps[atom])
        return float("inf")

    def valence(node: int) -> float:
        return float(capped[node].sum().item())

    # Reduce lowest-confidence bonds first until all atoms satisfy cap.
    order = sorted(range(len(pair_i)), key=lambda k: edge_conf[k])
    changed = True
    while changed:
        changed = False
        for k in order:
            i = pair_i[k]
            j = pair_j[k]
            if capped[i, j] <= 0:
                continue
            cap_i = cap(atom_types[i])
            cap_j = cap(atom_types[j])
            while capped[i, j] > 0 and (valence(i) > cap_i or valence(j) > cap_j):
                capped[i, j] -= 1.0
                capped[j, i] -= 1.0
                changed = True

    return capped.clamp(0, 3)


def _write_bond_class_probs(
    out_path: str,
    logits: torch.Tensor,
    data_cpu,
    atom_types: List[str],
    true_adj: torch.Tensor,
    pred_mat: torch.Tensor,
    cfg: ValidationConfig,
) -> None:
    probs = torch.softmax(logits, dim=-1).cpu()
    num_classes = int(probs.size(-1))
    topk = min(2, num_classes)
    topk_prob, topk_idx = torch.topk(probs, k=topk, dim=-1)

    header = [
        "i",
        "j",
        "atom_i",
        "atom_j",
        "react_bond",
        "true_bond",
        "top1_class",
        "top1_prob",
        "top2_class",
        "top2_prob",
        "correct_class",
        "correct_prob",
        "correct_rank",
        "postcap_pred_bond",
        "flag",
    ]
    if cfg.predict_bond_change:
        header.extend(["top1_delta", "top2_delta", "correct_delta"])
    for c in range(num_classes):
        header.append(f"p{c}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for k, (i, j) in enumerate(zip(data_cpu.pair_i.tolist(), data_cpu.pair_j.tolist())):
            pvals = probs[k].tolist()
            top1_class = int(topk_idx[k, 0].item())
            top1_prob = float(topk_prob[k, 0].item())
            if topk > 1:
                top2_class = int(topk_idx[k, 1].item())
                top2_prob = float(topk_prob[k, 1].item())
            else:
                top2_class = top1_class
                top2_prob = top1_prob

            correct_class = int(data_cpu.y_edge[k].item())
            correct_prob = float(pvals[correct_class])
            correct_rank = 1 + sum(1 for p in pvals if p > correct_prob)
            flag = "OK" if top1_class == correct_class else "ERROR"

            react_bond = float(data_cpu.react_adj[i, j].item())
            true_bond = float(true_adj[i, j].item())
            postcap_pred_bond = float(pred_mat[i, j].item())

            row = [
                str(i),
                str(j),
                atom_types[i],
                atom_types[j],
                f"{react_bond:.1f}",
                f"{true_bond:.1f}",
                str(top1_class),
                f"{top1_prob:.6f}",
                str(top2_class),
                f"{top2_prob:.6f}",
                str(correct_class),
                f"{correct_prob:.6f}",
                str(correct_rank),
                f"{postcap_pred_bond:.1f}",
                flag,
            ]

            if cfg.predict_bond_change:
                top1_delta = top1_class - cfg.delta_offset
                top2_delta = top2_class - cfg.delta_offset
                correct_delta = correct_class - cfg.delta_offset
                row.extend([str(top1_delta), str(top2_delta), str(correct_delta)])

            row.extend(f"{p:.6f}" for p in pvals)
            f.write("\t".join(row) + "\n")


def _reaction_by_id(reaction_id: int) -> Dict:
    for rxn in REACTIONS:
        if int(rxn.get("id", -1)) == int(reaction_id):
            return rxn
    return {}


def _write_rate_predictions(out_path: str, data_cpu, rate_pred_by_group: Optional[torch.Tensor]) -> Optional[float]:
    if not hasattr(data_cpu, "rate_target_by_group"):
        return None

    target_std = data_cpu.rate_target_by_group.float()
    if target_std.dim() == 1:
        target_std = target_std.view(1, -1)
    if hasattr(data_cpu, "rate_mask_by_group"):
        mask = data_cpu.rate_mask_by_group.float()
        if mask.dim() == 1:
            mask = mask.view(1, -1)
    else:
        mask = torch.ones_like(target_std)

    if hasattr(data_cpu, "rate_mean"):
        mean = data_cpu.rate_mean.float().view(1, -1)
    else:
        mean = torch.zeros((1, target_std.size(1)), dtype=torch.float32)
    if hasattr(data_cpu, "rate_std"):
        std = data_cpu.rate_std.float().view(1, -1)
    else:
        std = torch.ones((1, target_std.size(1)), dtype=torch.float32)

    if rate_pred_by_group is None:
        pred_std = torch.zeros_like(target_std)
    else:
        pred_std = rate_pred_by_group.float().cpu()
        if pred_std.dim() == 1:
            pred_std = pred_std.view(1, -1)
        n = min(pred_std.size(0), target_std.size(0))
        d = min(pred_std.size(1), target_std.size(1))
        pred_std = pred_std[:n, :d]
        target_std = target_std[:n, :d]
        mask = mask[:n, :d]
        mean = mean[:, :d]
        std = std[:, :d]

    target = target_std * std + mean
    pred = pred_std * std + mean

    denom = max(float(mask.sum().item()), 1.0)
    mae = float(((pred - target).abs() * mask).sum().item() / denom)

    reaction_ids = None
    if hasattr(data_cpu, "group_reaction_ids"):
        reaction_ids = data_cpu.group_reaction_ids.view(-1).tolist()
    elif hasattr(data_cpu, "reaction_id"):
        reaction_ids = [int(data_cpu.reaction_id)]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("group\treaction_id\trate_type\tfield\tpred_encoded\ttarget_encoded\tabs_err\tpred_physical\ttarget_physical\n")
        for g in range(target.size(0)):
            rid = int(reaction_ids[g]) if reaction_ids is not None and g < len(reaction_ids) else -1
            rxn = _reaction_by_id(rid)
            rate_type = str(rxn.get("rate", {}).get("type", "unknown"))

            p_main = pred[g, 0:3]
            t_main = target[g, 0:3]
            m_main = mask[g, 0:3]
            pA, pn, pEa = decode_rate_triplet(p_main)
            tA, tn, tEa = decode_rate_triplet(t_main)
            fields_main = [
                ("main_A", float(p_main[0].item()), float(t_main[0].item()), float(abs(p_main[0] - t_main[0]).item()), f"{pA:.6e}", f"{tA:.6e}", float(m_main[0].item())),
                ("main_n", float(p_main[1].item()), float(t_main[1].item()), float(abs(p_main[1] - t_main[1]).item()), f"{pn:.6f}", f"{tn:.6f}", float(m_main[1].item())),
                ("main_Ea", float(p_main[2].item()), float(t_main[2].item()), float(abs(p_main[2] - t_main[2]).item()), f"{pEa:.6f}", f"{tEa:.6f}", float(m_main[2].item())),
            ]
            for name, pe, te, ae, pp, tp, mm in fields_main:
                if mm > 0.0:
                    f.write(f"{g}\t{rid}\t{rate_type}\t{name}\t{pe:.6f}\t{te:.6f}\t{ae:.6f}\t{pp}\t{tp}\n")

            p_low = pred[g, 3:6]
            t_low = target[g, 3:6]
            m_low = mask[g, 3:6]
            pA2, pn2, pEa2 = decode_rate_triplet(p_low)
            tA2, tn2, tEa2 = decode_rate_triplet(t_low)
            fields_low = [
                ("low_A", float(p_low[0].item()), float(t_low[0].item()), float(abs(p_low[0] - t_low[0]).item()), f"{pA2:.6e}", f"{tA2:.6e}", float(m_low[0].item())),
                ("low_n", float(p_low[1].item()), float(t_low[1].item()), float(abs(p_low[1] - t_low[1]).item()), f"{pn2:.6f}", f"{tn2:.6f}", float(m_low[1].item())),
                ("low_Ea", float(p_low[2].item()), float(t_low[2].item()), float(abs(p_low[2] - t_low[2]).item()), f"{pEa2:.6f}", f"{tEa2:.6f}", float(m_low[2].item())),
            ]
            for name, pe, te, ae, pp, tp, mm in fields_low:
                if mm > 0.0:
                    f.write(f"{g}\t{rid}\t{rate_type}\t{name}\t{pe:.6f}\t{te:.6f}\t{ae:.6f}\t{pp}\t{tp}\n")
    return mae


def _reaction_subdir(bundle: DatasetBundle, idx: int, out_dir: str) -> str:
    reaction_id = bundle.eval_reaction_ids[idx] if idx < len(bundle.eval_reaction_ids) else (idx + 1)
    repeat_factor = bundle.eval_repeat_factors[idx] if idx < len(bundle.eval_repeat_factors) else 1
    branch_id = bundle.eval_branch_ids[idx] if idx < len(bundle.eval_branch_ids) else 0
    return os.path.join(out_dir, f"reaction_{reaction_id:02d}_b{branch_id}_x{repeat_factor}")


def _predict_one(model: EdgePredictor, data, atom_types: List[str], cfg: ValidationConfig):
    device = next(model.parameters()).device
    data_dev = data.to(device)
    with torch.no_grad():
        node_embed = model.encode(data_dev).cpu()
        out = model(data_dev, return_aux=True, sample_latent=getattr(model.cfg, "use_latent_branching", False))
        if isinstance(out, tuple):
            logits, aux = out
        else:
            logits, aux = out, {}
        pred_edges = logits.argmax(dim=-1).cpu()
        edge_conf = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
        rate_pred = aux.get("rate_pred_by_group")
        if rate_pred is not None:
            rate_pred = rate_pred.cpu()

    data_cpu = data_dev.cpu()
    logits_cpu = logits.cpu()
    pred_mat = _reconstruct_pred_matrix(data_cpu, pred_edges, cfg)

    if cfg.apply_valence_cap:
        pred_mat = _apply_valence_cap(
            pred_mat=pred_mat,
            atom_types=atom_types,
            pair_i=data_cpu.pair_i.tolist(),
            pair_j=data_cpu.pair_j.tolist(),
            edge_conf=edge_conf,
            valence_caps=cfg.valence_caps,
        )

    return data_cpu, logits_cpu, node_embed, pred_mat, rate_pred


def evaluate_snapshots_and_save_predicted_graphs(
    model: EdgePredictor,
    bundle: DatasetBundle,
    cfg: ValidationConfig,
    snapshots: Dict[int, Dict[str, torch.Tensor]],
) -> None:
    if not snapshots:
        return

    os.makedirs(cfg.out_dir, exist_ok=True)
    device = next(model.parameters()).device
    epoch_list = sorted(snapshots.keys())

    per_reaction_errors: Dict[str, List[Tuple[int, float]]] = {}

    print(f"[eval] running snapshot inference for epochs: {epoch_list}")
    for epoch in epoch_list:
        model.load_state_dict(snapshots[epoch])
        model.to(device)
        model.eval()

        for idx, data in enumerate(bundle.eval_data):
            atom_types = bundle.eval_atom_types[idx]
            true_adj = bundle.eval_true[idx]

            data_cpu, _, _, pred_mat, _ = _predict_one(model, data, atom_types, cfg)
            subdir = _reaction_subdir(bundle, idx, cfg.out_dir)
            os.makedirs(subdir, exist_ok=True)

            graph_out = os.path.join(subdir, f"graph_predicted_epoch_{epoch:04d}.png")
            plot_graph(pred_mat, atom_types, graph_out, f"Predicted (Epoch {epoch})")

            err = float((pred_mat - true_adj).abs().sum().item())
            per_reaction_errors.setdefault(subdir, []).append((epoch, err))

    for subdir, records in per_reaction_errors.items():
        records_sorted = sorted(records, key=lambda x: x[0])
        with open(os.path.join(subdir, "epoch_error_log.txt"), "w", encoding="utf-8") as f:
            f.write("epoch\tabs_error\n")
            for epoch, err in records_sorted:
                f.write(f"{epoch}\t{err:.1f}\n")


def evaluate_and_save(
    model: EdgePredictor,
    bundle: DatasetBundle,
    cfg: ValidationConfig,
) -> List[Tuple[str, float]]:
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    results: List[Tuple[str, float]] = []
    summary_lines: List[str] = []

    for idx, data in enumerate(bundle.eval_data):
        eqn = bundle.eval_labels[idx] if idx < len(bundle.eval_labels) else bundle.eval_equations[idx]
        true_adj = bundle.eval_true[idx]
        atom_types = bundle.eval_atom_types[idx]

        reaction_id = bundle.eval_reaction_ids[idx] if idx < len(bundle.eval_reaction_ids) else (idx + 1)
        repeat_factor = bundle.eval_repeat_factors[idx] if idx < len(bundle.eval_repeat_factors) else 1
        branch_id = bundle.eval_branch_ids[idx] if idx < len(bundle.eval_branch_ids) else 0
        branch_count = bundle.eval_branch_counts[idx] if idx < len(bundle.eval_branch_counts) else 1

        data_cpu, logits_cpu, node_embed, pred_mat, rate_pred_by_group = _predict_one(model, data, atom_types, cfg)

        subdir = _reaction_subdir(bundle, idx, cfg.out_dir)
        os.makedirs(subdir, exist_ok=True)

        with open(os.path.join(subdir, "pred_mat.txt"), "w", encoding="utf-8") as f:
            for row in pred_mat.tolist():
                f.write(" ".join(f"{v:.1f}" for v in row) + "\n")

        with open(os.path.join(subdir, "true_mat.txt"), "w", encoding="utf-8") as f:
            for row in true_adj.tolist():
                f.write(" ".join(f"{v:.1f}" for v in row) + "\n")

        with open(os.path.join(subdir, "reactant_adj.txt"), "w", encoding="utf-8") as f:
            for row in data_cpu.react_adj.tolist():
                f.write(" ".join(f"{v:.1f}" for v in row) + "\n")

        with open(os.path.join(subdir, "node_embeddings.txt"), "w", encoding="utf-8") as f:
            dim_headers = "\t".join(f"dim{d}" for d in range(node_embed.size(1)))
            f.write(f"node\tatom\t{dim_headers}\n")
            for node_idx in range(node_embed.size(0)):
                values = "\t".join(f"{v:.6f}" for v in node_embed[node_idx].tolist())
                f.write(f"{node_idx}\t{atom_types[node_idx]}\t{values}\n")

        _write_bond_class_probs(
            out_path=os.path.join(subdir, "bond_class_probs.txt"),
            logits=logits_cpu,
            data_cpu=data_cpu,
            atom_types=atom_types,
            true_adj=true_adj,
            pred_mat=pred_mat,
            cfg=cfg,
        )

        rate_mae = None
        if cfg.save_rate_predictions:
            rate_mae = _write_rate_predictions(
                out_path=os.path.join(subdir, "rate_coeffs.txt"),
                data_cpu=data_cpu,
                rate_pred_by_group=rate_pred_by_group,
            )

        err = float((pred_mat - true_adj).abs().sum().item())
        with open(os.path.join(subdir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(f"abs_error {err:.1f}\n")

        plot_graph(data_cpu.react_adj, atom_types, os.path.join(subdir, "graph_reactant.png"), "Reactant")
        plot_graph(pred_mat, atom_types, os.path.join(subdir, "graph_predicted.png"), "Predicted")
        plot_graph(true_adj, atom_types, os.path.join(subdir, "graph_target.png"), "Target")

        line = (
            f"Reaction {reaction_id:02d} b{branch_id}/{branch_count - 1} x{repeat_factor}: "
            f"abs_error {err:.1f} | rate_mae "
            f"{(f'{rate_mae:.4f}' if rate_mae is not None else 'n/a')} | {eqn}"
        )
        print(line)
        summary_lines.append(line)
        results.append((eqn, err))

    with open(os.path.join(cfg.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    return results
