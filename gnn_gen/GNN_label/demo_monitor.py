from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import os
import re

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import nn

from data_prep import DatasetBundle, DatasetConfig, build_reaction_data, decode_rate_triplet
from hydrogen_adjacency import REACTIONS, parse_equation, reactant_signature
from model import EdgePredictor
from plotting import plot_graph


@dataclass
class DemoMonitorConfig:
    out_dir: str
    reaction_id: int
    branch_id: Optional[int] = None  # optional override by branch slot in reactant group
    repeat_factor: Optional[int] = None
    predict_bond_change: bool = True
    delta_offset: int = 3
    index_scale: float = 1.0
    embedding_grid: Tuple[int, int] = (8, 8)
    # Preferred explicit ranges:
    node_embedding_vmin: Optional[float] = -10.0
    node_embedding_vmax: Optional[float] = 10.0
    decoder_vmin: Optional[float] = -20.0
    decoder_vmax: Optional[float] = 20.0
    # Backward-compatible symmetric limits (used when vmin/vmax are None).
    node_embedding_vlim: float = 10.0
    decoder_vlim: float = 20.0
    dpi: int = 220


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _has_third_body(equation: str) -> bool:
    return re.search(r"\bM\b", equation) is not None


def _reaction_index_by_id(reaction_id: int) -> int:
    for idx, rxn in enumerate(REACTIONS):
        if int(rxn.get("id", -1)) == int(reaction_id):
            return idx
    valid = [int(r["id"]) for r in REACTIONS]
    raise ValueError(f"Unknown reaction id {reaction_id}. Valid ids: {valid}")


def _reaction_group_members(reaction_index: int) -> List[int]:
    rxn = REACTIONS[reaction_index]
    reactants, _ = parse_equation(rxn["equation"])
    sig = reactant_signature(reactants)
    tb = _has_third_body(rxn["equation"])

    out: List[int] = []
    for idx, other in enumerate(REACTIONS):
        r2, _ = parse_equation(other["equation"])
        if reactant_signature(r2) == sig and _has_third_body(other["equation"]) == tb:
            out.append(idx)
    return out


def _build_demo_sample_from_base_reactions(
    cfg: DemoMonitorConfig,
):
    base_idx = _reaction_index_by_id(cfg.reaction_id)
    members = _reaction_group_members(base_idx)

    # Default: keep exact reaction id provided by user.
    selected_idx = base_idx
    if cfg.branch_id is not None:
        if cfg.branch_id < 0 or cfg.branch_id >= len(members):
            raise ValueError(
                f"branch_id {cfg.branch_id} out of range for reactant group size {len(members)} "
                f"(reaction id {cfg.reaction_id})."
            )
        selected_idx = members[cfg.branch_id]

    branch_count = len(members)
    branch_id = members.index(selected_idx)
    reaction_repeat = 1 if cfg.repeat_factor is None else max(1, int(cfg.repeat_factor))

    data_cfg = DatasetConfig(
        predict_bond_change=cfg.predict_bond_change,
        index_scale=cfg.index_scale,
    )
    data, _, atom_types = build_reaction_data(
        reaction_index=selected_idx,
        cfg=data_cfg,
        jitter=None,
        reaction_repeat=reaction_repeat,
        branch_id=branch_id,
        branch_count=branch_count,
    )
    reaction_id = int(REACTIONS[selected_idx]["id"])
    equation = str(REACTIONS[selected_idx]["equation"])
    return data, atom_types, reaction_id, branch_id, branch_count, reaction_repeat, equation


def _vector_to_grid(vec: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    rows, cols = shape
    size = rows * cols
    flat = vec.flatten()
    if flat.numel() >= size:
        out = flat[:size]
    else:
        pad = torch.zeros((size - flat.numel(),), dtype=flat.dtype, device=flat.device)
        out = torch.cat([flat, pad], dim=0)
    return out.view(rows, cols)


def _save_heatmap(mat: torch.Tensor, out_path: str, title: str, dpi: int) -> None:
    arr = mat.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    vmax = float(max(abs(arr.min()), abs(arr.max())))
    if vmax < 1e-12:
        vmax = 1.0
    im = ax.imshow(arr, cmap="coolwarm", aspect="equal", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("value", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_heatmap_with_fixed_scale(
    mat: torch.Tensor,
    out_path: str,
    title: str,
    dpi: int,
    vmin: float,
    vmax: float,
) -> None:
    arr = mat.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    lo = float(vmin)
    hi = float(vmax)
    if hi <= lo:
        hi = lo + 1e-6
    im = ax.imshow(arr, cmap="coolwarm", aspect="equal", vmin=lo, vmax=hi)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("value", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _resolve_range(vmin: Optional[float], vmax: Optional[float], vlim_fallback: float) -> Tuple[float, float]:
    if vmin is None and vmax is None:
        lim = max(abs(float(vlim_fallback)), 1e-12)
        return -lim, lim
    if vmin is None:
        lim = max(abs(float(vlim_fallback)), 1e-12)
        vmin = -lim
    if vmax is None:
        lim = max(abs(float(vlim_fallback)), 1e-12)
        vmax = lim
    lo = float(vmin)
    hi = float(vmax)
    if hi <= lo:
        raise ValueError(f"Invalid heatmap range: vmin={lo}, vmax={hi} (need vmax > vmin)")
    return lo, hi


def _decoder_intermediates(
    model: EdgePredictor,
    data,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      node_embed: [N, H]
      linear_outs[0]: first Linear output over candidate edges
      linear_outs[1]: second Linear output (logits) over candidate edges
    """
    x = model.encode(data)
    h_i = x[data.pair_i]
    h_j = x[data.pair_j]
    edge_group = model._edge_group_ids(data, num_edges=h_i.size(0), device=h_i.device)
    pair_feat = torch.cat([h_i, h_j], dim=-1)

    if model.cfg.molecule_balanced_pool:
        g_edge = model._edge_group_pooled_feature(x=x, data=data, edge_group=edge_group)
        pair_feat = torch.cat([pair_feat, g_edge], dim=-1)

    pair_feat = torch.cat([pair_feat, data.react_edge.view(-1, 1)], dim=-1)

    if model.cfg.use_branch_feature:
        if model.cfg.branch_feature_mode == "scalar":
            b = model._edge_group_scalar_feature(
                data=data,
                edge_group=edge_group,
                group_field_name="branch_value_by_group",
                legacy_field_name="branch_value",
                device=h_i.device,
            )
        else:
            b = model._contextual_branch_feature(x=x, data=data, edge_group=edge_group, device=h_i.device)
        pair_feat = torch.cat([pair_feat, b], dim=-1)

    if model.cfg.use_third_body_feature:
        t = model._edge_group_scalar_feature(
            data=data,
            edge_group=edge_group,
            group_field_name="third_body_by_group",
            legacy_field_name="third_body_value",
            device=h_i.device,
        )
        pair_feat = torch.cat([pair_feat, t], dim=-1)

    if model.cfg.use_latent_branching:
        z_edge, _, _ = model._latent_edge_feature(
            x=x,
            data=data,
            edge_group=edge_group,
            sample_latent=False,
        )
        pair_feat = torch.cat([pair_feat, z_edge], dim=-1)

    linear_outs: List[torch.Tensor] = []
    z = pair_feat
    for layer in model.edge_mlp:
        z = layer(z)
        if isinstance(layer, nn.Linear):
            linear_outs.append(z)

    if not linear_outs:
        raise RuntimeError("edge_mlp has no Linear layers to visualize.")
    if len(linear_outs) == 1:
        linear_outs = [linear_outs[0], linear_outs[0]]
    return x, linear_outs[0], linear_outs[1]


def _bond_form_probability(
    logits: torch.Tensor,
    react_edge: torch.Tensor,
    predict_bond_change: bool,
    delta_offset: int,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    num_classes = int(probs.size(-1))
    if predict_bond_change:
        cls = torch.arange(num_classes, device=logits.device, dtype=torch.float32).view(1, -1)
        delta = cls - float(delta_offset)
        final_bond = torch.clamp(react_edge.view(-1, 1) + delta, min=0.0, max=3.0)
        mask = (final_bond > 0.0).float()
        return (probs * mask).sum(dim=-1)
    if num_classes <= 1:
        return torch.zeros((probs.size(0),), dtype=torch.float32, device=logits.device)
    return probs[:, 1:].sum(dim=-1)


def _plot_confidence_complete_graph(
    atom_types: List[str],
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    p_form: torch.Tensor,
    out_path: str,
    title: str,
    dpi: int,
) -> None:
    n = len(atom_types)
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(i, atom=atom_types[i])
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j)

    pos = nx.spring_layout(graph, seed=42)
    labels = {i: f"{atom_types[i]}_{i}" for i in range(n)}
    colors = ["#d32f2f" if atom_types[i] == "O" else "#1976d2" for i in range(n)]

    plt.figure(figsize=(7.2, 7.2))
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=500)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)

    for k, (i, j) in enumerate(zip(pair_i.tolist(), pair_j.tolist())):
        p = float(p_form[k].item())
        alpha = 0.05 + 0.95 * max(0.0, min(1.0, p))
        width = 0.2 + 3.8 * max(0.0, min(1.0, p))
        color = "#1b5e20" if p >= 0.5 else "#9e9e9e"
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(i, j)],
            width=width,
            alpha=alpha,
            edge_color=color,
        )

    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _de_standardize_rate(pred_std: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    m = mean.view(1, -1).float()
    s = std.view(1, -1).float()
    d = min(pred_std.size(1), m.size(1), s.size(1))
    out = pred_std.clone().float()
    out[:, :d] = out[:, :d] * s[:, :d] + m[:, :d]
    return out


def _build_rate_lines(
    rate_pred_by_group: Optional[torch.Tensor],
    data_sample,
    rate_mean: Optional[torch.Tensor],
    rate_std: Optional[torch.Tensor],
) -> List[str]:
    lines: List[str] = []
    if rate_pred_by_group is None:
        return ["rate head: unavailable"]
    if not hasattr(data_sample, "rate_target_by_group"):
        return ["rate target: unavailable"]

    pred = rate_pred_by_group.detach().cpu().float()
    if pred.dim() == 1:
        pred = pred.view(1, -1)
    target = data_sample.rate_target_by_group.detach().cpu().float()
    if target.dim() == 1:
        target = target.view(1, -1)
    if hasattr(data_sample, "rate_mask_by_group"):
        mask = data_sample.rate_mask_by_group.detach().cpu().float()
        if mask.dim() == 1:
            mask = mask.view(1, -1)
    else:
        mask = torch.ones_like(target)

    n = min(pred.size(0), target.size(0), 1)
    d = min(pred.size(1), target.size(1))
    pred = pred[:n, :d]
    target = target[:n, :d]
    mask = mask[:n, :d]

    # De-standardize predictions when stats are provided by training bundle.
    if rate_mean is not None and rate_std is not None:
        pred_raw = _de_standardize_rate(pred, rate_mean.detach().cpu(), rate_std.detach().cpu())
    else:
        pred_raw = pred
    target_raw = target

    lines.append("Rate coefficients (group 0)")
    lines.append("field        pred        target      abs_err")

    p_main = pred_raw[0, 0:3]
    t_main = target_raw[0, 0:3]
    m_main = mask[0, 0:3]
    if float(m_main.sum().item()) > 0:
        pA, pn, pEa = decode_rate_triplet(p_main)
        tA, tn, tEa = decode_rate_triplet(t_main)
        lines.append(f"main_A  {pA:10.3e} {tA:10.3e} {abs(pA - tA):10.3e}")
        lines.append(f"main_n  {pn:10.4f} {tn:10.4f} {abs(pn - tn):10.4f}")
        lines.append(f"main_Ea {pEa:10.3f} {tEa:10.3f} {abs(pEa - tEa):10.3f}")

    p_low = pred_raw[0, 3:6]
    t_low = target_raw[0, 3:6]
    m_low = mask[0, 3:6]
    if float(m_low.sum().item()) > 0:
        pA2, pn2, pEa2 = decode_rate_triplet(p_low)
        tA2, tn2, tEa2 = decode_rate_triplet(t_low)
        lines.append(f"low_A   {pA2:10.3e} {tA2:10.3e} {abs(pA2 - tA2):10.3e}")
        lines.append(f"low_n   {pn2:10.4f} {tn2:10.4f} {abs(pn2 - tn2):10.4f}")
        lines.append(f"low_Ea  {pEa2:10.3f} {tEa2:10.3f} {abs(pEa2 - tEa2):10.3f}")
    return lines


def _extract_main_rate_triplet(
    rate_pred_by_group: Optional[torch.Tensor],
    data_sample,
    rate_mean: Optional[torch.Tensor],
    rate_std: Optional[torch.Tensor],
) -> Optional[Dict[str, List[float]]]:
    if rate_pred_by_group is None:
        return None
    if not hasattr(data_sample, "rate_target_by_group"):
        return None

    pred = rate_pred_by_group.detach().cpu().float()
    if pred.dim() == 1:
        pred = pred.view(1, -1)
    target = data_sample.rate_target_by_group.detach().cpu().float()
    if target.dim() == 1:
        target = target.view(1, -1)
    if hasattr(data_sample, "rate_mask_by_group"):
        mask = data_sample.rate_mask_by_group.detach().cpu().float()
        if mask.dim() == 1:
            mask = mask.view(1, -1)
    else:
        mask = torch.ones_like(target)

    n = min(pred.size(0), target.size(0), 1)
    d = min(pred.size(1), target.size(1))
    if n <= 0 or d < 3:
        return None
    pred = pred[:n, :d]
    target = target[:n, :d]
    mask = mask[:n, :d]

    if float(mask[0, 0:3].sum().item()) <= 0.0:
        return None

    if rate_mean is not None and rate_std is not None:
        pred_raw = _de_standardize_rate(pred, rate_mean.detach().cpu(), rate_std.detach().cpu())
    else:
        pred_raw = pred
    target_raw = target

    pA, pn, pEa = decode_rate_triplet(pred_raw[0, 0:3])
    tA, tn, tEa = decode_rate_triplet(target_raw[0, 0:3])
    pred_vals = [float(pA), float(pn), float(pEa)]
    target_vals = [float(tA), float(tn), float(tEa)]
    abs_err = [abs(a - b) for a, b in zip(pred_vals, target_vals)]
    scale = [max(abs(a), abs(b), 1e-12) for a, b in zip(pred_vals, target_vals)]
    norm_err = [e / s for e, s in zip(abs_err, scale)]
    return {
        "labels": ["A", "n", "Ea"],
        "pred": pred_vals,
        "target": target_vals,
        "abs_err": abs_err,
        "scale": scale,
        "norm_err": norm_err,
    }


def _save_convergence_plot(
    losses: Optional[List[float]],
    epoch: int,
    out_path: str,
    dpi: int,
) -> None:
    if losses and len(losses) > 0:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        xs = list(range(1, len(losses) + 1))
        ax.plot(xs, losses, color="#1565c0", linewidth=1.4)
        e = min(max(1, int(epoch)), len(losses))
        y = float(losses[e - 1])
        ax.scatter([e], [y], color="red", s=44, zorder=3, label=f"checkpoint epoch {epoch}")
        ax.axvline(e, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title("Training Convergence")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(loc="best", fontsize=8, frameon=False)
    else:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(0.5, 0.5, "Loss history unavailable", ha="center", va="center")
        ax.set_title("Training Convergence")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_rate_error_radar(
    triplet: Optional[Dict[str, List[float]]],
    out_path: str,
    dpi: int,
) -> None:
    if triplet is None:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        ax.text(0.5, 0.5, "Rate coefficients unavailable", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return

    labels = triplet["labels"]
    pred_vals = triplet["pred"]
    target_vals = triplet["target"]
    # Target-referenced scaling:
    # - target is fixed at 1.0 on every axis
    # - prediction is |pred| / |target|
    # - clip prediction at axis max (2.0)
    axis_max = 2.0
    target_norm = [1.0 for _ in labels]
    pred_norm = [
        min(axis_max, abs(float(p)) / max(abs(float(t)), 1e-12))
        for p, t in zip(pred_vals, target_vals)
    ]

    angles = [2.0 * math.pi * i / len(labels) for i in range(len(labels))]
    angles += angles[:1]
    pred_norm_closed = pred_norm + pred_norm[:1]
    target_norm_closed = target_norm + target_norm[:1]

    fig = plt.figure(figsize=(4.6, 4.0))
    ax_radar = fig.add_subplot(1, 1, 1, polar=True)

    ax_radar.plot(angles, target_norm_closed, color="#1565c0", linewidth=2.2, label="target")
    ax_radar.fill(angles, target_norm_closed, color="#90caf9", alpha=0.30)
    ax_radar.plot(angles, pred_norm_closed, color="#c62828", linewidth=2.2, label="pred")
    ax_radar.fill(angles, pred_norm_closed, color="#ef9a9a", alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=12)
    ax_radar.set_ylim(0.0, axis_max)
    ax_radar.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax_radar.set_yticklabels(["0.5", "1.0", "1.5", "2.0"], fontsize=10)
    ax_radar.set_title("Rate Coefficients Radar (Target-Normalized)", fontsize=15, pad=12)
    ax_radar.legend(loc="lower right", fontsize=12, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run_single_reaction_demo(
    model: EdgePredictor,
    bundle: DatasetBundle,
    snapshots: Dict[int, Dict[str, torch.Tensor]],
    cfg: DemoMonitorConfig,
    losses: Optional[List[float]] = None,
) -> None:
    if not snapshots:
        raise ValueError("No snapshots found. Enable snapshot saving in training config.")

    demo_root = cfg.out_dir
    _ensure_dir(demo_root)

    # Demo sample is selected from original 21 reactions, not eval split.
    _ = bundle  # kept in signature for compatibility with caller
    data_sample, atom_types, reaction_id, branch_id, branch_count, repeat_factor, equation = _build_demo_sample_from_base_reactions(cfg)

    sample_dir = os.path.join(demo_root, f"reaction_{reaction_id:02d}_b{branch_id}_x{repeat_factor}")
    _ensure_dir(sample_dir)

    device = next(model.parameters()).device
    epoch_list = sorted(snapshots.keys())
    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    with open(os.path.join(sample_dir, "demo_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"reaction_id={reaction_id}\n")
        f.write(f"branch_id={branch_id}\n")
        f.write(f"branch_count={branch_count}\n")
        f.write(f"repeat_factor={repeat_factor}\n")
        f.write(f"equation={equation}\n")
        f.write(f"snapshots={epoch_list}\n")

    # Save reactant graph once for this demo sample.
    plot_graph(
        data_sample.react_adj.detach().cpu(),
        atom_types,
        os.path.join(sample_dir, "graph_reactant.png"),
        f"Reactant Graph (Reaction {reaction_id})",
    )

    node_embed_vmin, node_embed_vmax = _resolve_range(
        cfg.node_embedding_vmin, cfg.node_embedding_vmax, cfg.node_embedding_vlim
    )
    decoder_vmin, decoder_vmax = _resolve_range(
        cfg.decoder_vmin, cfg.decoder_vmax, cfg.decoder_vlim
    )

    with open(os.path.join(sample_dir, "demo_info.txt"), "a", encoding="utf-8") as f:
        f.write(f"node_embedding_fixed_vmin={node_embed_vmin:.6f}\n")
        f.write(f"node_embedding_fixed_vmax={node_embed_vmax:.6f}\n")
        f.write(f"decoder_fixed_vmin={decoder_vmin:.6f}\n")
        f.write(f"decoder_fixed_vmax={decoder_vmax:.6f}\n")

    for epoch in epoch_list:
        model.load_state_dict(snapshots[epoch])
        model.to(device)
        model.eval()

        with torch.no_grad():
            data_dev = data_sample.to(device)
            node_embed, dec_l1, dec_l2 = _decoder_intermediates(model, data_dev)
            logits = dec_l2
            p_form = _bond_form_probability(
                logits=logits,
                react_edge=data_dev.react_edge,
                predict_bond_change=cfg.predict_bond_change,
                delta_offset=cfg.delta_offset,
            )
            out = model(data_dev, return_aux=True, sample_latent=False)
            if isinstance(out, tuple):
                _, aux = out
            else:
                aux = {}
            rate_pred = aux.get("rate_pred_by_group")
            if rate_pred is not None:
                rate_pred = rate_pred.detach().cpu()

        ep_dir = os.path.join(sample_dir, f"epoch_{epoch:04d}")
        _ensure_dir(ep_dir)

        # Node embedding heatmaps: each node as 8x8 grid (with pad/truncation).
        for n_idx in range(node_embed.size(0)):
            grid = _vector_to_grid(node_embed[n_idx], cfg.embedding_grid)
            _save_heatmap_with_fixed_scale(
                grid,
                os.path.join(ep_dir, f"node_embed_{n_idx:02d}.png"),
                f"Epoch {epoch} Node {n_idx} ({atom_types[n_idx]})",
                dpi=cfg.dpi,
                vmin=node_embed_vmin,
                vmax=node_embed_vmax,
            )

        # Edge decoder layer heatmaps (mean over candidate edges -> 8x8).
        l1_mean_grid = _vector_to_grid(dec_l1.mean(dim=0), cfg.embedding_grid)
        l2_mean_grid = _vector_to_grid(dec_l2.mean(dim=0), cfg.embedding_grid)
        _save_heatmap_with_fixed_scale(
            l1_mean_grid,
            os.path.join(ep_dir, "edge_decoder_layer1_mean.png"),
            f"Epoch {epoch} Edge Decoder Layer1 Mean",
            dpi=cfg.dpi,
            vmin=decoder_vmin,
            vmax=decoder_vmax,
        )
        _save_heatmap_with_fixed_scale(
            l2_mean_grid,
            os.path.join(ep_dir, "edge_decoder_layer2_mean.png"),
            f"Epoch {epoch} Edge Decoder Layer2 Mean (Logits)",
            dpi=cfg.dpi,
            vmin=decoder_vmin,
            vmax=decoder_vmax,
        )

        # Confidence-weighted complete graph over all candidate edges.
        _plot_confidence_complete_graph(
            atom_types=atom_types,
            pair_i=data_dev.pair_i.detach().cpu(),
            pair_j=data_dev.pair_j.detach().cpu(),
            p_form=p_form.detach().cpu(),
            out_path=os.path.join(ep_dir, "graph_confidence_complete.png"),
            title=f"Epoch {epoch} Bond Confidence (Complete Graph)",
            dpi=cfg.dpi,
        )

        # Save raw confidence table.
        with open(os.path.join(ep_dir, "bond_confidence.txt"), "w", encoding="utf-8") as f:
            f.write("i\tj\tatom_i\tatom_j\tp_form\n")
            for i, j, p in zip(
                data_dev.pair_i.detach().cpu().tolist(),
                data_dev.pair_j.detach().cpu().tolist(),
                p_form.detach().cpu().tolist(),
            ):
                f.write(f"{i}\t{j}\t{atom_types[i]}\t{atom_types[j]}\t{float(p):.6f}\n")

        rate_lines = _build_rate_lines(
            rate_pred_by_group=rate_pred,
            data_sample=data_sample,
            rate_mean=getattr(bundle, "rate_mean", None),
            rate_std=getattr(bundle, "rate_std", None),
        )
        with open(os.path.join(ep_dir, "rate_prediction.txt"), "w", encoding="utf-8") as f:
            for line in rate_lines:
                f.write(line + "\n")

        _save_convergence_plot(
            losses=losses,
            epoch=epoch,
            out_path=os.path.join(ep_dir, "convergence_checkpoint.png"),
            dpi=cfg.dpi,
        )

        triplet = _extract_main_rate_triplet(
            rate_pred_by_group=rate_pred,
            data_sample=data_sample,
            rate_mean=getattr(bundle, "rate_mean", None),
            rate_std=getattr(bundle, "rate_std", None),
        )
        _save_rate_error_radar(
            triplet=triplet,
            out_path=os.path.join(ep_dir, "rate_error_radar.png"),
            dpi=cfg.dpi,
        )

    model.load_state_dict(original_state)
    model.to(device)
