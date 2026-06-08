from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import re

import torch
from torch_geometric.data import Data

from hydrogen_adjacency import (
    REACTIONS,
    MOLECULES,
    parse_equation,
    combine_species,
    reorder_to_target,
    reactant_signature,
)


@dataclass
class DatasetConfig:
    predict_bond_change: bool = True
    selected_reactions: Optional[List[int]] = None  # reaction ids (1..21)
    filter_unique_reactants: bool = True
    perturb_training: bool = True
    perturb_samples: int = 5
    perturb_range: float = 0.1
    include_base_sample: bool = True
    copy_counts: Optional[List[int]] = None  # e.g. [2, 3, 4]
    random_group_training: bool = False
    random_group_train_samples: int = 100
    random_group_min_size: int = 1
    random_group_max_size: int = 5
    random_group_seed: int = 0
    random_group_training_replace_base: bool = False
    random_group_eval: bool = False
    random_group_eval_samples: int = 30
    random_group_eval_min_size: int = 1
    random_group_eval_max_size: int = 5
    random_group_eval_seed: int = 1
    random_group_eval_replace_base: bool = False
    ensure_disjoint_train_eval: bool = True
    index_scale: float = 1.0
    # Atom mapping mode for product alignment:
    # True  -> signature-seeded + local-swap refinement to minimize bond edits
    # False -> stable type-FIFO mapping only
    use_min_edit_atom_mapping: bool = True
    # Reaction-level mapping uncertainty:
    # apply exactly this many random same-type swaps per reaction mapping
    # (shared by all derived samples of that reaction).
    mapping_uncertainty_swaps: int = 0
    # Seed for deterministic reaction-level mapping perturbation.
    mapping_uncertainty_seed: int = 0
    rate_standardize: bool = True
    show_progress: bool = True


@dataclass
class DatasetBundle:
    train_data: List[Data]
    eval_data: List[Data]
    eval_true: List[torch.Tensor]
    eval_atom_types: List[List[str]]
    eval_equations: List[str]
    eval_labels: List[str]
    eval_reaction_ids: List[int]
    eval_repeat_factors: List[int]
    eval_branch_ids: List[int]
    eval_branch_counts: List[int]
    used_equations: List[str]
    omitted_equations: List[str]
    rate_mean: torch.Tensor
    rate_std: torch.Tensor


def num_classes_and_offset(predict_bond_change: bool) -> Tuple[int, int]:
    if predict_bond_change:
        return 7, 3  # delta in [-3, 3]
    return 4, 0  # bond order in [0, 3]


RATE_TARGET_DIM = 6  # [logA_main, n_main, Ea_main, logA_low, n_low, Ea_low]
RATE_LOGA_SCALE = 25.0
RATE_N_SCALE = 5.0
RATE_EA_SCALE = 20000.0
RATE_A_MIN = 1e-30


def _encode_rate_triplet(A: float, n: float, Ea: float) -> torch.Tensor:
    logA = torch.log10(torch.tensor(max(float(A), RATE_A_MIN), dtype=torch.float32))
    return torch.tensor(
        [
            float(logA.item()) / RATE_LOGA_SCALE,
            float(n) / RATE_N_SCALE,
            float(Ea) / RATE_EA_SCALE,
        ],
        dtype=torch.float32,
    )


def decode_rate_triplet(encoded_triplet: torch.Tensor) -> Tuple[float, float, float]:
    t = encoded_triplet.float().view(-1)
    if t.numel() != 3:
        raise ValueError("decode_rate_triplet expects 3 encoded values.")
    logA = float(t[0].item()) * RATE_LOGA_SCALE
    n = float(t[1].item()) * RATE_N_SCALE
    Ea = float(t[2].item()) * RATE_EA_SCALE
    A = float(10.0 ** logA)
    return A, n, Ea


def _reaction_rate_target_and_mask(rxn: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes per-reaction rate coefficients into fixed-size target + mask tensors.
    Target layout (size 6):
      [main_logA, main_n, main_Ea, low_logA, low_n, low_Ea]
    Mask indicates valid coefficients:
      - Arrhenius / three-body: first 3 valid.
      - Falloff: all 6 valid.
    """
    target = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)
    mask = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)

    rate = rxn.get("rate", {})
    rtype = str(rate.get("type", "")).lower()

    if rtype in {"arrhenius", "three_body"}:
        main = _encode_rate_triplet(rate.get("A", 0.0), rate.get("n", 0.0), rate.get("Ea", 0.0))
        target[0:3] = main
        mask[0:3] = 1.0
    elif rtype == "falloff":
        high = rate.get("high", {})
        low = rate.get("low", {})
        target[0:3] = _encode_rate_triplet(high.get("A", 0.0), high.get("n", 0.0), high.get("Ea", 0.0))
        target[3:6] = _encode_rate_triplet(low.get("A", 0.0), low.get("n", 0.0), low.get("Ea", 0.0))
        mask[:] = 1.0

    return target, mask


def _compute_rate_standardization(data_list: List[Data]) -> Tuple[torch.Tensor, torch.Tensor]:
    sum_vec = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)
    count_vec = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)

    for d in data_list:
        if not hasattr(d, "rate_target_by_group"):
            continue
        t = d.rate_target_by_group.float()
        if t.dim() == 1:
            t = t.view(1, -1)
        if hasattr(d, "rate_mask_by_group"):
            m = d.rate_mask_by_group.float()
            if m.dim() == 1:
                m = m.view(1, -1)
        else:
            m = torch.ones_like(t)

        d_used = min(t.size(1), RATE_TARGET_DIM)
        if d_used <= 0:
            continue
        sum_vec[:d_used] += (t[:, :d_used] * m[:, :d_used]).sum(dim=0)
        count_vec[:d_used] += m[:, :d_used].sum(dim=0)

    mean = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)
    valid = count_vec > 0
    mean[valid] = sum_vec[valid] / count_vec[valid]

    sq_sum = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)
    for d in data_list:
        if not hasattr(d, "rate_target_by_group"):
            continue
        t = d.rate_target_by_group.float()
        if t.dim() == 1:
            t = t.view(1, -1)
        if hasattr(d, "rate_mask_by_group"):
            m = d.rate_mask_by_group.float()
            if m.dim() == 1:
                m = m.view(1, -1)
        else:
            m = torch.ones_like(t)

        d_used = min(t.size(1), RATE_TARGET_DIM)
        if d_used <= 0:
            continue
        diff = t[:, :d_used] - mean[:d_used].view(1, -1)
        sq_sum[:d_used] += (diff * diff * m[:, :d_used]).sum(dim=0)

    var = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)
    var[valid] = sq_sum[valid] / torch.clamp(count_vec[valid], min=1.0)
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    std[~valid] = 1.0
    std = torch.clamp(std, min=1e-4)
    return mean, std


def _apply_rate_standardization(data_list: List[Data], mean: torch.Tensor, std: torch.Tensor) -> None:
    mean = mean.view(1, -1)
    std = std.view(1, -1)
    for d in data_list:
        if not hasattr(d, "rate_target_by_group"):
            continue
        t = d.rate_target_by_group.float()
        if t.dim() == 1:
            t = t.view(1, -1)
        if hasattr(d, "rate_mask_by_group"):
            m = d.rate_mask_by_group.float()
            if m.dim() == 1:
                m = m.view(1, -1)
        else:
            m = torch.ones_like(t)

        d_used = min(t.size(1), mean.size(1))
        if d_used <= 0:
            continue
        t_std = t.clone()
        t_std[:, :d_used] = ((t[:, :d_used] - mean[:, :d_used]) / std[:, :d_used]) * m[:, :d_used]
        d.rate_target_by_group = t_std
        d.rate_mean = mean.squeeze(0).clone()
        d.rate_std = std.squeeze(0).clone()


def _has_third_body(equation: str) -> bool:
    # Detect bare M and (+ M) notation as third-body participation.
    return re.search(r"\bM\b", equation) is not None


def _molecule_local_indices(
    reactants: List[str],
) -> Tuple[List[str], List[int], List[int], List[int]]:
    """
    Atom-wise metadata in combine_species order:
    - atom type
    - molecule index within each species type
    - atom index within that molecule
    - molecule graph index inside this reaction side
    """
    mol_count_by_species: Dict[str, int] = {}
    atom_types: List[str] = []
    molecule_idx: List[int] = []
    atom_in_molecule_idx: List[int] = []
    molecule_graph_idx: List[int] = []
    graph_idx = 0

    for species in reactants:
        mol_idx = mol_count_by_species.get(species, 0)
        mol_count_by_species[species] = mol_idx + 1
        for local_idx, atom in enumerate(MOLECULES[species].atom_types):
            atom_types.append(atom)
            molecule_idx.append(mol_idx)
            atom_in_molecule_idx.append(local_idx)
            molecule_graph_idx.append(graph_idx)
        graph_idx += 1

    return atom_types, molecule_idx, atom_in_molecule_idx, molecule_graph_idx


def _build_node_features(
    atom_types: List[str],
    molecule_idx: List[int],
    atom_in_molecule_idx: List[int],
    index_scale: float,
    jitter: Optional[float] = None,
) -> torch.Tensor:
    """
    Per-atom feature:
    [is_H, is_O, molecule_index_within_species, atom_index_within_molecule]
    """
    rows: List[List[float]] = []
    for atom, mol_i, local_i in zip(atom_types, molecule_idx, atom_in_molecule_idx):
        mol_val = float(mol_i)
        local_val = float(local_i)
        if jitter is not None:
            mol_val += random.uniform(-jitter, jitter)
            local_val += random.uniform(-jitter, jitter)
        mol_val *= index_scale
        local_val *= index_scale
        if atom == "H":
            rows.append([1.0, 0.0, mol_val, local_val])
        else:
            rows.append([0.0, 1.0, mol_val, local_val])
    return torch.tensor(rows, dtype=torch.float32)


def _branch_value(branch_id: int, branch_count: int) -> float:
    if branch_count <= 1:
        return 0.0
    return float(branch_id) / float(branch_count - 1)


def _permute_square_adjacency(adj: List[List[int]], perm: List[int]) -> List[List[int]]:
    size = len(adj)
    return [[adj[perm[i]][perm[j]] for j in range(size)] for i in range(size)]


def _apply_reaction_mapping_uncertainty(
    prod_adj: List[List[int]],
    atom_types: List[str],
    noise_swaps: int,
    rng: random.Random,
) -> List[List[int]]:
    """
    Perturb aligned product mapping by random same-type swaps.
    This preserves atom-type constraints while introducing mapping noise.
    """
    n = len(atom_types)
    if noise_swaps <= 0 or n < 2:
        return prod_adj

    idx_by_type: Dict[str, List[int]] = {}
    for idx, atom in enumerate(atom_types):
        idx_by_type.setdefault(atom, []).append(idx)
    eligible_by_type: Dict[str, List[int]] = {
        atom: idxs for atom, idxs in idx_by_type.items() if len(idxs) >= 2
    }
    if not eligible_by_type:
        return prod_adj

    perm = list(range(n))
    eligible_types = sorted(eligible_by_type.keys())
    for _ in range(int(noise_swaps)):
        atom = rng.choice(eligible_types)
        idxs = eligible_by_type[atom]
        i, j = rng.sample(idxs, 2)
        perm[i], perm[j] = perm[j], perm[i]
    return _permute_square_adjacency(prod_adj, perm)


def _reaction_noise_seed(
    base_seed: int,
    reaction_index: int,
) -> int:
    """
    Build a stable deterministic seed for reaction-level mapping perturbation.
    """
    parts = (
        int(base_seed),
        int(reaction_index),
    )
    x = 0x6A09E667
    for p in parts:
        x = ((x ^ (p & 0xFFFFFFFF)) * 0x45D9F3B) & 0xFFFFFFFF
        x = ((x ^ (x >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
        x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return int(x)


def build_reaction_data(
    reaction_index: int,
    cfg: DatasetConfig,
    jitter: Optional[float] = None,
    reaction_repeat: int = 1,
    branch_id: int = 0,
    branch_count: int = 1,
) -> Tuple[Data, torch.Tensor, List[str]]:
    num_classes, delta_offset = num_classes_and_offset(cfg.predict_bond_change)

    rxn = REACTIONS[reaction_index]
    reactants, products = parse_equation(rxn["equation"])
    rate_target, rate_mask = _reaction_rate_target_and_mask(rxn)
    if reaction_repeat < 1:
        raise ValueError("reaction_repeat must be >= 1")
    reactants_use = reactants * reaction_repeat
    products_use = products * reaction_repeat

    react_adj, react_atoms = combine_species(reactants_use)
    prod_adj, prod_atoms = combine_species(products_use)
    if len(react_atoms) != len(prod_atoms):
        raise ValueError(f"Atom count mismatch in reaction '{rxn['equation']}'")

    target_adj_for_mapping = react_adj if cfg.use_min_edit_atom_mapping else None
    prod_adj = reorder_to_target(
        target_atoms=react_atoms,
        source_atoms=prod_atoms,
        source_adj=prod_adj,
        target_adj=target_adj_for_mapping,
    )
    if cfg.mapping_uncertainty_swaps > 0:
        rng = random.Random(
            _reaction_noise_seed(
                base_seed=cfg.mapping_uncertainty_seed,
                reaction_index=reaction_index,
            )
        )
        prod_adj = _apply_reaction_mapping_uncertainty(
            prod_adj=prod_adj,
            atom_types=react_atoms,
            noise_swaps=cfg.mapping_uncertainty_swaps,
            rng=rng,
        )

    ordered_atoms, molecule_idx, atom_in_molecule_idx, molecule_graph_idx = _molecule_local_indices(
        reactants_use
    )
    if ordered_atoms != react_atoms:
        raise ValueError("Reactant atom ordering mismatch while building features.")

    n = len(react_atoms)
    pair_index = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_i = torch.tensor([i for i, _ in pair_index], dtype=torch.long)
    pair_j = torch.tensor([j for _, j in pair_index], dtype=torch.long)

    x = _build_node_features(
        react_atoms,
        molecule_idx=molecule_idx,
        atom_in_molecule_idx=atom_in_molecule_idx,
        index_scale=cfg.index_scale,
        jitter=jitter,
    )

    edges: List[List[int]] = []
    edge_attrs: List[List[float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            bond = react_adj[i][j]
            if bond != 0:
                edges.append([i, j])
                edge_attrs.append([float(bond)])
                edges.append([j, i])
                edge_attrs.append([float(bond)])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    react_edge = torch.tensor([react_adj[i][j] for i, j in pair_index], dtype=torch.float32)
    react_adj_t = torch.tensor(react_adj, dtype=torch.float32)
    prod_adj_t = torch.tensor(prod_adj, dtype=torch.float32)

    if cfg.predict_bond_change:
        y_edge = (prod_adj_t[pair_i, pair_j] - react_adj_t[pair_i, pair_j] + delta_offset).long()
    else:
        y_edge = prod_adj_t[pair_i, pair_j].long().clamp(0, num_classes - 1)

    third_body_flag = 1.0 if _has_third_body(rxn["equation"]) else 0.0

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.n_nodes = n
    data.pair_i = pair_i
    data.pair_j = pair_j
    data.react_edge = react_edge
    data.react_adj = react_adj_t
    data.y_edge = y_edge
    data.atom_types = react_atoms
    data.mol_id = torch.tensor(molecule_graph_idx, dtype=torch.long)
    data.reaction_equation = rxn["equation"]
    data.reaction_id = rxn["id"]
    data.reaction_repeat = reaction_repeat

    # Group-aware conditioning tensors (currently single-group per sample).
    # These are used by the model to support future mixed multi-reaction samples
    # where each edge can condition on its own reactant-group branch/third-body.
    data.node_group_id = torch.zeros(n, dtype=torch.long)
    data.pair_group_id = torch.zeros(len(pair_index), dtype=torch.long)
    data.branch_id_by_group = torch.tensor([branch_id], dtype=torch.long)
    data.branch_count_by_group = torch.tensor([branch_count], dtype=torch.long)
    data.branch_value_by_group = torch.tensor([_branch_value(branch_id, branch_count)], dtype=torch.float32)
    data.third_body_by_group = torch.tensor([third_body_flag], dtype=torch.float32)

    # Legacy single-global conditioning tensors kept for compatibility.
    data.branch_id = torch.tensor([branch_id], dtype=torch.long)
    data.branch_count = torch.tensor([branch_count], dtype=torch.long)
    data.branch_value = torch.tensor([_branch_value(branch_id, branch_count)], dtype=torch.float32)
    data.third_body_value = torch.tensor([third_body_flag], dtype=torch.float32)
    data.rate_target_by_group = rate_target.view(1, -1)
    data.rate_mask_by_group = rate_mask.view(1, -1)
    data.group_reaction_ids = torch.tensor([rxn["id"]], dtype=torch.long)
    return data, prod_adj_t, react_atoms


def build_random_packed_reaction_data(
    reaction_indices: List[int],
    cfg: DatasetConfig,
    sig_groups: Dict["GroupKey", List[int]],
    index_to_sig: Dict[int, "GroupKey"],
    jitter: Optional[float] = None,
    reaction_repeat: int = 1,
) -> Tuple[Data, torch.Tensor, List[str]]:
    """
    Pack an arbitrary list of reactions into one disconnected graph sample.
    Each packed block keeps its own branch metadata relative to its reactant group.
    """
    if not reaction_indices:
        raise ValueError("reaction_indices must be non-empty")

    block_data: List[Data] = []
    block_true: List[torch.Tensor] = []
    block_atoms: List[List[str]] = []
    block_equations: List[str] = []
    block_reaction_ids: List[int] = []
    block_branch_ids: List[int] = []
    block_branch_counts: List[int] = []

    for rxn_idx in reaction_indices:
        sig = index_to_sig[rxn_idx]
        branch_group = sig_groups[sig]
        branch_count = len(branch_group)
        branch_id = branch_group.index(rxn_idx)

        d, true_adj, atoms = build_reaction_data(
            reaction_index=rxn_idx,
            cfg=cfg,
            jitter=jitter,
            reaction_repeat=reaction_repeat,
            branch_id=branch_id,
            branch_count=branch_count,
        )
        block_data.append(d)
        block_true.append(true_adj)
        block_atoms.append(atoms)
        block_equations.append(REACTIONS[rxn_idx]["equation"])
        block_reaction_ids.append(REACTIONS[rxn_idx]["id"])
        block_branch_ids.append(branch_id)
        block_branch_counts.append(branch_count)

    x_list: List[torch.Tensor] = []
    edge_index_list: List[torch.Tensor] = []
    edge_attr_list: List[torch.Tensor] = []
    pair_i_list: List[torch.Tensor] = []
    pair_j_list: List[torch.Tensor] = []
    react_edge_list: List[torch.Tensor] = []
    y_edge_list: List[torch.Tensor] = []
    atom_types_all: List[str] = []
    mol_id_list: List[torch.Tensor] = []
    node_group_list: List[torch.Tensor] = []
    pair_group_list: List[torch.Tensor] = []
    react_adj_blocks: List[torch.Tensor] = []
    true_adj_blocks: List[torch.Tensor] = []
    third_body_group_vals: List[float] = []
    rate_target_by_group: List[torch.Tensor] = []
    rate_mask_by_group: List[torch.Tensor] = []

    node_offset = 0
    mol_offset = 0
    for g, (d, true_adj, atoms) in enumerate(zip(block_data, block_true, block_atoms)):
        n = int(d.n_nodes)
        x_list.append(d.x)
        atom_types_all.extend(atoms)

        if d.edge_index.numel() > 0:
            edge_index_list.append(d.edge_index + node_offset)
            edge_attr_list.append(d.edge_attr)

        pair_i_list.append(d.pair_i + node_offset)
        pair_j_list.append(d.pair_j + node_offset)
        react_edge_list.append(d.react_edge)
        y_edge_list.append(d.y_edge)

        if hasattr(d, "mol_id") and d.mol_id.numel() > 0:
            mol_local = d.mol_id.long()
            mol_id_list.append(mol_local + mol_offset)
            mol_offset += int(mol_local.max().item()) + 1
        else:
            mol_id_list.append(torch.empty((0,), dtype=torch.long))

        node_group_list.append(torch.full((n,), g, dtype=torch.long))
        pair_group_list.append(torch.full((d.pair_i.numel(),), g, dtype=torch.long))

        react_adj_blocks.append(d.react_adj)
        true_adj_blocks.append(true_adj)

        if hasattr(d, "third_body_by_group") and d.third_body_by_group.numel() > 0:
            third_body_group_vals.append(float(d.third_body_by_group.view(-1)[0].item()))
        elif hasattr(d, "third_body_value") and d.third_body_value.numel() > 0:
            third_body_group_vals.append(float(d.third_body_value.view(-1)[0].item()))
        else:
            third_body_group_vals.append(0.0)

        if hasattr(d, "rate_target_by_group") and d.rate_target_by_group.numel() > 0:
            rate_target_by_group.append(d.rate_target_by_group.view(1, -1))
        else:
            rate_target_by_group.append(torch.zeros((1, RATE_TARGET_DIM), dtype=torch.float32))
        if hasattr(d, "rate_mask_by_group") and d.rate_mask_by_group.numel() > 0:
            rate_mask_by_group.append(d.rate_mask_by_group.view(1, -1))
        else:
            rate_mask_by_group.append(torch.zeros((1, RATE_TARGET_DIM), dtype=torch.float32))

        node_offset += n

    x = torch.cat(x_list, dim=0)
    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    pair_i = torch.cat(pair_i_list, dim=0)
    pair_j = torch.cat(pair_j_list, dim=0)
    react_edge = torch.cat(react_edge_list, dim=0)
    y_edge = torch.cat(y_edge_list, dim=0)
    mol_id = torch.cat(mol_id_list, dim=0) if mol_id_list else torch.empty((0,), dtype=torch.long)
    node_group_id = torch.cat(node_group_list, dim=0)
    pair_group_id = torch.cat(pair_group_list, dim=0)

    react_adj = torch.block_diag(*react_adj_blocks)
    true_adj_all = torch.block_diag(*true_adj_blocks)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.n_nodes = x.size(0)
    data.pair_i = pair_i
    data.pair_j = pair_j
    data.react_edge = react_edge
    data.react_adj = react_adj
    data.y_edge = y_edge
    data.atom_types = atom_types_all
    data.mol_id = mol_id

    data.node_group_id = node_group_id
    data.pair_group_id = pair_group_id
    data.branch_id_by_group = torch.tensor(block_branch_ids, dtype=torch.long)
    data.branch_count_by_group = torch.tensor(block_branch_counts, dtype=torch.long)
    data.branch_value_by_group = torch.tensor(
        [_branch_value(bid, bcnt) for bid, bcnt in zip(block_branch_ids, block_branch_counts)],
        dtype=torch.float32,
    )
    data.third_body_by_group = torch.tensor(third_body_group_vals, dtype=torch.float32)
    data.rate_target_by_group = torch.cat(rate_target_by_group, dim=0)
    data.rate_mask_by_group = torch.cat(rate_mask_by_group, dim=0)
    data.group_reaction_ids = torch.tensor(block_reaction_ids, dtype=torch.long)

    data.reaction_equation = " || ".join(block_equations)
    data.reaction_id = block_reaction_ids[0]
    data.reaction_ids = torch.tensor(block_reaction_ids, dtype=torch.long)
    data.reaction_repeat = reaction_repeat

    # Legacy single-global tensors kept for compatibility.
    data.branch_id = torch.tensor([block_branch_ids[0]], dtype=torch.long)
    data.branch_count = torch.tensor([block_branch_counts[0]], dtype=torch.long)
    data.branch_value = torch.tensor([_branch_value(block_branch_ids[0], block_branch_counts[0])], dtype=torch.float32)
    data.third_body_value = torch.tensor([third_body_group_vals[0]], dtype=torch.float32)
    return data, true_adj_all, atom_types_all


GroupKey = Tuple[Tuple[Tuple[str, int], ...], bool]


def _build_reactant_groups(candidate_indices: List[int]) -> Dict[GroupKey, List[int]]:
    groups: Dict[GroupKey, List[int]] = {}
    for idx in candidate_indices:
        eqn = REACTIONS[idx]["equation"]
        reactants, _ = parse_equation(eqn)
        sig = reactant_signature(reactants)
        key: GroupKey = (sig, _has_third_body(eqn))
        groups.setdefault(key, []).append(idx)
    return groups


def _resolve_copy_counts(cfg: DatasetConfig) -> List[int]:
    if cfg.copy_counts is None:
        return [1]

    values = sorted({int(v) for v in cfg.copy_counts if int(v) >= 1})
    if not values:
        raise ValueError("copy_counts must contain at least one integer >= 1")
    return values


def _sample_key(data: Data) -> Tuple[Tuple[int, ...], int]:
    """
    Semantic key used for train/eval disjoint filtering.
    We compare by reaction-id set and reaction_repeat, independent of node order.
    """
    rep = int(getattr(data, "reaction_repeat", 1))
    if hasattr(data, "reaction_ids"):
        ids = tuple(sorted(int(v) for v in data.reaction_ids.view(-1).tolist()))
    elif hasattr(data, "reaction_id"):
        ids = (int(data.reaction_id),)
    else:
        ids = tuple()
    return ids, rep


def build_datasets(cfg: DatasetConfig) -> DatasetBundle:
    train_data: List[Data] = []
    eval_data: List[Data] = []
    eval_true: List[torch.Tensor] = []
    eval_atom_types: List[List[str]] = []
    eval_equations: List[str] = []
    eval_labels: List[str] = []
    eval_reaction_ids: List[int] = []
    eval_repeat_factors: List[int] = []
    eval_branch_ids: List[int] = []
    eval_branch_counts: List[int] = []
    used_equations: List[str] = []
    omitted_equations: List[str] = []

    selected = set(cfg.selected_reactions) if cfg.selected_reactions is not None else None
    candidate_indices = [i for i, rxn in enumerate(REACTIONS) if selected is None or rxn["id"] in selected]
    total_candidates = len(candidate_indices)

    if cfg.show_progress:
        print(f"[data] building dataset from {total_candidates} candidate reactions")

    copy_values = _resolve_copy_counts(cfg)

    sig_groups = _build_reactant_groups(candidate_indices)
    index_to_sig: Dict[int, GroupKey] = {}
    for sig, idxs in sig_groups.items():
        for idx in idxs:
            index_to_sig[idx] = sig

    seen_signatures = set()
    processed = 0

    for idx in candidate_indices:
        processed += 1
        rxn = REACTIONS[idx]
        rid = rxn["id"]

        sig = index_to_sig[idx]
        group = sig_groups[sig]

        if cfg.filter_unique_reactants:
            if sig in seen_signatures:
                omitted_equations.append(rxn["equation"])
                if cfg.show_progress:
                    print(f"[data {processed}/{total_candidates}] skip R{rid:02d} (duplicate reactants)")
                continue
            seen_signatures.add(sig)
            branch_variants = [(0, 1)]
        else:
            # Branching enabled for all reactions when unique-reactant filter is off.
            # - If multiple products share this (reactant signature, third-body flag):
            #   use branch index per reaction.
            # - If only one product exists: keep a single branch id 0.
            if len(group) == 1:
                # Non-branching reactants: keep a single branch id 0.
                branch_variants = [(0, 1)]
            else:
                branch_id = group.index(idx)
                branch_variants = [(branch_id, len(group))]

        used_equations.append(rxn["equation"])

        eval_before = len(eval_data)
        train_before = len(train_data)

        eval_repeat_values = copy_values
        train_repeat_values = copy_values

        for branch_id, branch_count in branch_variants:
            for rep in eval_repeat_values:
                eval_sample, true_adj, atom_types = build_reaction_data(
                    idx,
                    cfg=cfg,
                    reaction_repeat=rep,
                    branch_id=branch_id,
                    branch_count=branch_count,
                )
                eval_data.append(eval_sample)
                eval_true.append(true_adj)
                eval_atom_types.append(atom_types)
                eval_equations.append(rxn["equation"])
                eval_reaction_ids.append(rid)
                eval_repeat_factors.append(rep)
                eval_branch_ids.append(branch_id)
                eval_branch_counts.append(branch_count)
                suffix = f"[b{branch_id}/{branch_count - 1}]"
                if rep == 1:
                    eval_labels.append(f"{rxn['equation']} {suffix}")
                else:
                    eval_labels.append(f"{rxn['equation']} {suffix} [x{rep}]")

            for rep in train_repeat_values:
                base_sample, _, _ = build_reaction_data(
                    idx,
                    cfg=cfg,
                    reaction_repeat=rep,
                    branch_id=branch_id,
                    branch_count=branch_count,
                )
                if cfg.perturb_training:
                    if cfg.include_base_sample:
                        train_data.append(base_sample)
                    for _ in range(cfg.perturb_samples):
                        aug_sample, _, _ = build_reaction_data(
                            idx,
                            cfg=cfg,
                            reaction_repeat=rep,
                            branch_id=branch_id,
                            branch_count=branch_count,
                            jitter=cfg.perturb_range,
                        )
                        train_data.append(aug_sample)
                else:
                    train_data.append(base_sample)

        if cfg.show_progress:
            added_eval = len(eval_data) - eval_before
            added_train = len(train_data) - train_before
            print(
                f"[data {processed}/{total_candidates}] keep R{rid:02d}: "
                f"branches={len(branch_variants)} +eval {added_eval}, +train {added_train}"
            )

    if cfg.random_group_training:
        source_indices = list(candidate_indices)
        if not source_indices:
            raise RuntimeError("No candidate reactions available for random group training.")

        min_k = max(1, int(cfg.random_group_min_size))
        max_k = max(min_k, int(cfg.random_group_max_size))
        max_k = min(max_k, len(source_indices))
        min_k = min(min_k, max_k)
        n_samples = max(1, int(cfg.random_group_train_samples))
        rng = random.Random(cfg.random_group_seed)

        random_train: List[Data] = []
        if cfg.show_progress:
            print(
                f"[data] random group training enabled: samples={n_samples}, "
                f"group_size_range=[{min_k},{max_k}]"
            )

        for s in range(1, n_samples + 1):
            group_size = rng.randint(min_k, max_k)
            chosen = rng.sample(source_indices, k=group_size)

            for rep in copy_values:
                base_sample, _, _ = build_random_packed_reaction_data(
                    reaction_indices=chosen,
                    cfg=cfg,
                    sig_groups=sig_groups,
                    index_to_sig=index_to_sig,
                    reaction_repeat=rep,
                )
                if cfg.perturb_training:
                    if cfg.include_base_sample:
                        random_train.append(base_sample)
                    for _ in range(cfg.perturb_samples):
                        aug_sample, _, _ = build_random_packed_reaction_data(
                            reaction_indices=chosen,
                            cfg=cfg,
                            sig_groups=sig_groups,
                            index_to_sig=index_to_sig,
                            reaction_repeat=rep,
                            jitter=cfg.perturb_range,
                        )
                        random_train.append(aug_sample)
                else:
                    random_train.append(base_sample)

            if cfg.show_progress and (s == 1 or s == n_samples or s % max(1, n_samples // 10) == 0):
                print(f"[data random {s}/{n_samples}] packed {group_size} reactions")

        if cfg.random_group_training_replace_base:
            train_data = random_train
        else:
            train_data.extend(random_train)

    if cfg.random_group_eval:
        source_indices = list(candidate_indices)
        if not source_indices:
            raise RuntimeError("No candidate reactions available for random group evaluation.")

        min_k = max(1, int(cfg.random_group_eval_min_size))
        max_k = max(min_k, int(cfg.random_group_eval_max_size))
        max_k = min(max_k, len(source_indices))
        min_k = min(min_k, max_k)
        n_samples = max(1, int(cfg.random_group_eval_samples))
        rng = random.Random(cfg.random_group_eval_seed)

        random_eval_data: List[Data] = []
        random_eval_true: List[torch.Tensor] = []
        random_eval_atom_types: List[List[str]] = []
        random_eval_equations: List[str] = []
        random_eval_labels: List[str] = []
        random_eval_reaction_ids: List[int] = []
        random_eval_repeat_factors: List[int] = []
        random_eval_branch_ids: List[int] = []
        random_eval_branch_counts: List[int] = []

        if cfg.show_progress:
            print(
                f"[data] random group eval enabled: samples={n_samples}, "
                f"group_size_range=[{min_k},{max_k}]"
            )

        eval_uid = 1000
        for s in range(1, n_samples + 1):
            group_size = rng.randint(min_k, max_k)
            chosen = rng.sample(source_indices, k=group_size)
            chosen_eqs = [REACTIONS[i]["equation"] for i in chosen]
            eq_join = " || ".join(chosen_eqs)

            for rep in copy_values:
                sample, true_adj, atom_types = build_random_packed_reaction_data(
                    reaction_indices=chosen,
                    cfg=cfg,
                    sig_groups=sig_groups,
                    index_to_sig=index_to_sig,
                    reaction_repeat=rep,
                )
                random_eval_data.append(sample)
                random_eval_true.append(true_adj)
                random_eval_atom_types.append(atom_types)
                random_eval_equations.append(eq_join)

                eval_uid += 1
                random_eval_reaction_ids.append(eval_uid)
                random_eval_repeat_factors.append(rep)
                random_eval_branch_ids.append(0)
                random_eval_branch_counts.append(1)

                if rep == 1:
                    random_eval_labels.append(f"RANDOM[{s:03d}] k{group_size} :: {eq_join}")
                else:
                    random_eval_labels.append(f"RANDOM[{s:03d}] k{group_size} [x{rep}] :: {eq_join}")

            if cfg.show_progress and (s == 1 or s == n_samples or s % max(1, n_samples // 10) == 0):
                print(f"[data random-eval {s}/{n_samples}] packed {group_size} reactions")

        if cfg.random_group_eval_replace_base:
            eval_data = random_eval_data
            eval_true = random_eval_true
            eval_atom_types = random_eval_atom_types
            eval_equations = random_eval_equations
            eval_labels = random_eval_labels
            eval_reaction_ids = random_eval_reaction_ids
            eval_repeat_factors = random_eval_repeat_factors
            eval_branch_ids = random_eval_branch_ids
            eval_branch_counts = random_eval_branch_counts
        else:
            eval_data.extend(random_eval_data)
            eval_true.extend(random_eval_true)
            eval_atom_types.extend(random_eval_atom_types)
            eval_equations.extend(random_eval_equations)
            eval_labels.extend(random_eval_labels)
            eval_reaction_ids.extend(random_eval_reaction_ids)
            eval_repeat_factors.extend(random_eval_repeat_factors)
            eval_branch_ids.extend(random_eval_branch_ids)
            eval_branch_counts.extend(random_eval_branch_counts)

    if cfg.ensure_disjoint_train_eval:
        train_keys = {_sample_key(d) for d in train_data}
        keep_idx: List[int] = []
        dropped = 0
        for i, d in enumerate(eval_data):
            if _sample_key(d) in train_keys:
                dropped += 1
            else:
                keep_idx.append(i)

        if dropped > 0:
            eval_data = [eval_data[i] for i in keep_idx]
            eval_true = [eval_true[i] for i in keep_idx]
            eval_atom_types = [eval_atom_types[i] for i in keep_idx]
            eval_equations = [eval_equations[i] for i in keep_idx]
            eval_labels = [eval_labels[i] for i in keep_idx]
            eval_reaction_ids = [eval_reaction_ids[i] for i in keep_idx]
            eval_repeat_factors = [eval_repeat_factors[i] for i in keep_idx]
            eval_branch_ids = [eval_branch_ids[i] for i in keep_idx]
            eval_branch_counts = [eval_branch_counts[i] for i in keep_idx]
            if cfg.show_progress:
                print(f"[data] disjoint filter removed {dropped} overlapping eval samples")

    if cfg.show_progress:
        print(
            f"[data] done: train={len(train_data)} eval={len(eval_data)} "
            f"used={len(used_equations)} omitted={len(omitted_equations)}"
        )

    rate_mean = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)
    rate_std = torch.ones((RATE_TARGET_DIM,), dtype=torch.float32)
    if cfg.rate_standardize and train_data:
        rate_mean, rate_std = _compute_rate_standardization(train_data)
        _apply_rate_standardization(train_data, rate_mean, rate_std)
        _apply_rate_standardization(eval_data, rate_mean, rate_std)
        if cfg.show_progress:
            mean_str = ", ".join(f"{v:.4f}" for v in rate_mean.tolist())
            std_str = ", ".join(f"{v:.4f}" for v in rate_std.tolist())
            print(f"[data] rate standardization mean=[{mean_str}]")
            print(f"[data] rate standardization std=[{std_str}]")

    return DatasetBundle(
        train_data=train_data,
        eval_data=eval_data,
        eval_true=eval_true,
        eval_atom_types=eval_atom_types,
        eval_equations=eval_equations,
        eval_labels=eval_labels,
        eval_reaction_ids=eval_reaction_ids,
        eval_repeat_factors=eval_repeat_factors,
        eval_branch_ids=eval_branch_ids,
        eval_branch_counts=eval_branch_counts,
        used_equations=used_equations,
        omitted_equations=omitted_equations,
        rate_mean=rate_mean,
        rate_std=rate_std,
    )
