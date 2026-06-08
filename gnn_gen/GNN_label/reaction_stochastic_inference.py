from __future__ import annotations

import itertools
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch_geometric.data import Data

from hydrogen_adjacency import MoleculeGraph, MOLECULES, REACTIONS, combine_species, parse_equation, reactant_signature
from data_prep import decode_rate_triplet
from model import EdgePredictor, ModelConfig
from plotting import plot_graph


# Checkpoint / runtime
MODEL_PATH = "/scratch.raptor/du000298/GNN_label/reaction_dataset_prediction_min_edits_4_swap/model.pt"
OUT_DIR = "reaction_stochastic_inference_min_edits_4_swap"
DEVICE = "cuda"
SEED = 0

# Initial pool and simulation controls
INITIAL_POOL: Dict[str, int] = {"H2": 15, "O2": 15}
NUM_EVENTS = 600
P_BIMOLECULAR = 0.9
TEMPERATURE = 300.0  # K

# Third-body stochastic model
PRESSURE = 0.1
PRESSURE_REF = 1.0

# If True, allow inference channels for reactant signatures not present in
# the template reaction map by using a single default branch (id=0,count=1).
ALLOW_UNSEEN_REACTANT_CHANNELS = True

# Channel selection over all candidate reaction channels:
# - "argmax": deterministic top-1 (previous behavior)
# - "sample": draw from softmax(log_prop) (current stochastic behavior)
CHANNEL_SELECTION_MODE = "sample"

# Kinetics constants / numerics
R_GAS_CAL = 1.98720425864083  # cal/(mol*K), consistent with Ea in cal/mol
RATE_TARGET_DIM = 6
RATE_LOGA_SCALE = 25.0
RATE_N_SCALE = 5.0
RATE_EA_SCALE = 20000.0
EPS = 1e-12

# Latent-branch inference (used only if checkpoint model_cfg.use_latent_branching=True)
LATENT_SAMPLES_PER_EVENT = 4
# "best_confidence": keep only highest-confidence latent sample per reactant channel
# "random": keep all latent samples as separate channels for propensity competition
LATENT_SAMPLE_SELECTION = "random"

# Inference postprocess
APPLY_VALENCE_CAP = True
VALENCE_CAPS = {"H": 1, "O": 2}
# Penalize no-bond-change channels in log-propensity:
# log_prop <- log_prop - NO_BOND_CHANGE_LOG_PENALTY if predicted adjacency equals reactant adjacency.
NO_BOND_CHANGE_LOG_PENALTY = 1.0
# Save registered unknown species graph artifacts under OUT_DIR/<UNKNOWN_SPECIES_DIR_NAME>/.
UNKNOWN_SPECIES_DIR_NAME = "unknown_species"
# If True: unmatched predicted components are rejected.
# If False: unmatched components are auto-registered as synthetic species UNK_####.
ENFORCE_KNOWN_SPECIES_MAPPING = False

# Animation output
SAVE_OVERALL_GIF = True
OVERALL_GIF_NAME = "overall_steps.gif"
OVERALL_GIF_DURATION_MS = 450

# Save / logging frequency controls
# 1 => save every step
SAVE_EVENT_ARTIFACTS_EVERY = 5
SAVE_OVERALL_PLOTS_EVERY = 5
ALWAYS_SAVE_FINAL_STEP = True
# Disable per-event local graph PNGs by default (reduces IO and runtime).
SAVE_EVENT_LOCAL_GRAPHS = False
# 0 => disable progress print
PRINT_PROGRESS_EVERY = 5


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_cfg(raw_cfg: Dict[str, Any]) -> ModelConfig:
    valid = {f.name for f in fields(ModelConfig)}
    kwargs = {k: v for k, v in raw_cfg.items() if k in valid}
    return ModelConfig(**kwargs)


def _has_third_body(equation: str) -> bool:
    return re.search(r"\bM\b", equation) is not None


def _build_branch_template_map() -> Dict[Tuple[Tuple[Tuple[str, int], ...], bool], List[int]]:
    groups: Dict[Tuple[Tuple[Tuple[str, int], ...], bool], List[int]] = {}
    for rxn in REACTIONS:
        reactants, _ = parse_equation(rxn["equation"])
        key = (reactant_signature(reactants), _has_third_body(rxn["equation"]))
        groups.setdefault(key, []).append(rxn["id"])
    for key in groups:
        groups[key] = sorted(groups[key])
    return groups


def _build_reaction_map_by_id() -> Dict[int, Dict]:
    return {int(r["id"]): r for r in REACTIONS}


def _encode_rate_triplet(A: float, n: float, Ea: float) -> torch.Tensor:
    A_clamped = max(float(A), 1e-30)
    return torch.tensor(
        [
            math.log10(A_clamped) / RATE_LOGA_SCALE,
            float(n) / RATE_N_SCALE,
            float(Ea) / RATE_EA_SCALE,
        ],
        dtype=torch.float32,
    )


def _rate_vector_from_template(rxn: Dict) -> torch.Tensor:
    out = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)
    rate = rxn.get("rate", {})
    rtype = str(rate.get("type", "")).lower()
    if rtype in {"arrhenius", "three_body"}:
        out[0:3] = _encode_rate_triplet(rate.get("A", 0.0), rate.get("n", 0.0), rate.get("Ea", 0.0))
    elif rtype == "falloff":
        high = rate.get("high", {})
        low = rate.get("low", {})
        out[0:3] = _encode_rate_triplet(high.get("A", 0.0), high.get("n", 0.0), high.get("Ea", 0.0))
        out[3:6] = _encode_rate_triplet(low.get("A", 0.0), low.get("n", 0.0), low.get("Ea", 0.0))
    return out


def _rate_vector_from_aux(
    aux: Dict[str, torch.Tensor],
    rate_mean: torch.Tensor,
    rate_std: torch.Tensor,
) -> Optional[torch.Tensor]:
    pred = aux.get("rate_pred_by_group")
    if pred is None:
        return None
    p = pred.detach().float().cpu()
    if p.dim() == 1:
        p = p.view(1, -1)
    if p.numel() == 0:
        return None
    vec = p[0]
    d = min(vec.numel(), rate_mean.numel(), RATE_TARGET_DIM)
    out = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)
    out[:d] = vec[:d] * rate_std[:d] + rate_mean[:d]
    return out


def _logsumexp_two(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _arrhenius_logk_from_encoded(encoded_triplet: torch.Tensor, temperature: float) -> float:
    e = encoded_triplet.float().view(-1)
    if e.numel() != 3:
        return float("-inf")
    log10A = float(e[0].item()) * RATE_LOGA_SCALE
    n = float(e[1].item()) * RATE_N_SCALE
    Ea = float(e[2].item()) * RATE_EA_SCALE
    lnA = log10A * math.log(10.0)
    return lnA + n * math.log(max(temperature, EPS)) - Ea / (R_GAS_CAL * max(temperature, EPS))


def _reaction_rate_type_hint(rxn: Optional[Dict], third_body_flag: bool) -> str:
    if rxn is not None:
        rt = str(rxn.get("rate", {}).get("type", "")).lower()
        if rt:
            return rt
    return "three_body" if third_body_flag else "arrhenius"


def _log_k_effective(
    encoded_rate_vec: torch.Tensor,
    rate_type: str,
    temperature: float,
    pressure: float,
    pressure_ref: float,
) -> float:
    logk_main = _arrhenius_logk_from_encoded(encoded_rate_vec[0:3], temperature)
    if not math.isfinite(logk_main):
        return float("-inf")

    m_eff = max(float(pressure) / max(float(pressure_ref), EPS), EPS)
    log_m = math.log(m_eff)
    rt = rate_type.lower()
    if rt == "three_body":
        return logk_main + log_m
    if rt == "falloff":
        logk_low = _arrhenius_logk_from_encoded(encoded_rate_vec[3:6], temperature)
        if not math.isfinite(logk_low):
            return logk_main
        logk_lowm = logk_low + log_m
        # k = (k_inf * k0[M]) / (k_inf + k0[M])
        return logk_main + logk_lowm - _logsumexp_two(logk_main, logk_lowm)
    return logk_main


def _reactant_count_factor(reactants: Sequence[str], pool: Dict[str, int]) -> float:
    if not reactants:
        return 0.0
    if len(reactants) == 1:
        return float(pool.get(reactants[0], 0))
    if len(reactants) == 2:
        a, b = reactants[0], reactants[1]
        if a == b:
            n = float(pool.get(a, 0))
            return max(0.0, n * (n - 1.0) * 0.5)
        return float(pool.get(a, 0)) * float(pool.get(b, 0))
    return 0.0


def _enumerate_collision_groups(pool: Dict[str, int]) -> List[List[str]]:
    species = sorted([s for s, c in pool.items() if int(c) > 0])
    out: List[List[str]] = []
    for s in species:
        out.append([s])
    for i, a in enumerate(species):
        for b in species[i:]:
            if a == b and int(pool.get(a, 0)) < 2:
                continue
            out.append([a, b])
    return out

def _molecule_local_indices(reactants: Sequence[str]) -> Tuple[List[str], List[int], List[int], List[int]]:
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
    atom_types: Sequence[str],
    molecule_idx: Sequence[int],
    atom_in_molecule_idx: Sequence[int],
    index_scale: float = 1.0,
) -> torch.Tensor:
    rows: List[List[float]] = []
    for atom, mol_i, local_i in zip(atom_types, molecule_idx, atom_in_molecule_idx):
        mol_val = float(mol_i) * index_scale
        local_val = float(local_i) * index_scale
        if atom == "H":
            rows.append([1.0, 0.0, mol_val, local_val])
        else:
            rows.append([0.0, 1.0, mol_val, local_val])
    return torch.tensor(rows, dtype=torch.float32)


def _branch_value(branch_id: int, branch_count: int) -> float:
    if branch_count <= 1:
        return 0.0
    return float(branch_id) / float(branch_count - 1)


def _build_inference_data(
    reactants: Sequence[str],
    branch_id: int,
    branch_count: int,
    third_body_flag: float,
    index_scale: float = 1.0,
) -> Tuple[Data, List[str]]:
    react_adj, react_atoms = combine_species(list(reactants))
    atom_types, molecule_idx, atom_in_molecule_idx, mol_graph_idx = _molecule_local_indices(reactants)
    if atom_types != react_atoms:
        raise ValueError("Reactant atom ordering mismatch in inference builder.")

    n = len(react_atoms)
    pair_index = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_i = torch.tensor([i for i, _ in pair_index], dtype=torch.long)
    pair_j = torch.tensor([j for _, j in pair_index], dtype=torch.long)

    x = _build_node_features(
        atom_types=react_atoms,
        molecule_idx=molecule_idx,
        atom_in_molecule_idx=atom_in_molecule_idx,
        index_scale=index_scale,
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

    react_adj_t = torch.tensor(react_adj, dtype=torch.float32)
    react_edge = torch.tensor([react_adj[i][j] for i, j in pair_index], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.n_nodes = n
    data.pair_i = pair_i
    data.pair_j = pair_j
    data.react_adj = react_adj_t
    data.react_edge = react_edge
    data.atom_types = list(react_atoms)
    data.mol_id = torch.tensor(mol_graph_idx, dtype=torch.long)

    # Single-group inference sample.
    data.node_group_id = torch.zeros(n, dtype=torch.long)
    data.pair_group_id = torch.zeros(len(pair_index), dtype=torch.long)
    data.branch_id_by_group = torch.tensor([branch_id], dtype=torch.long)
    data.branch_count_by_group = torch.tensor([branch_count], dtype=torch.long)
    data.branch_value_by_group = torch.tensor([_branch_value(branch_id, branch_count)], dtype=torch.float32)
    data.third_body_by_group = torch.tensor([third_body_flag], dtype=torch.float32)

    # Legacy compatibility fields.
    data.branch_id = torch.tensor([branch_id], dtype=torch.long)
    data.branch_count = torch.tensor([branch_count], dtype=torch.long)
    data.branch_value = torch.tensor([_branch_value(branch_id, branch_count)], dtype=torch.float32)
    data.third_body_value = torch.tensor([third_body_flag], dtype=torch.float32)
    data.reaction_repeat = 1
    return data, list(react_atoms)


def _reconstruct_pred_matrix(data, pred_edges: torch.Tensor, predict_bond_change: bool, delta_offset: int) -> torch.Tensor:
    n = int(data.n_nodes)
    pred_mat = torch.zeros((n, n), dtype=torch.float32)
    for k, (i, j) in enumerate(zip(data.pair_i.tolist(), data.pair_j.tolist())):
        if predict_bond_change:
            delta = float(pred_edges[k].item() - delta_offset)
            pred_mat[i, j] = data.react_adj[i, j] + delta
            pred_mat[j, i] = data.react_adj[j, i] + delta
        else:
            bond = float(pred_edges[k].item())
            pred_mat[i, j] = bond
            pred_mat[j, i] = bond
    return pred_mat.clamp(0, 3)


def _no_bond_change_stats(
    react_adj: torch.Tensor,
    pred_mat: torch.Tensor,
) -> Tuple[bool, int]:
    n = int(min(react_adj.size(0), pred_mat.size(0)))
    if n <= 1:
        return True, 0
    iu = torch.triu_indices(n, n, offset=1)
    react = react_adj[iu[0], iu[1]].round().to(torch.int64).cpu()
    pred = pred_mat[iu[0], iu[1]].round().to(torch.int64).cpu()
    changed = int((react != pred).sum().item())
    return changed == 0, changed


def _apply_valence_cap(
    pred_mat: torch.Tensor,
    atom_types: Sequence[str],
    pair_i: Sequence[int],
    pair_j: Sequence[int],
    edge_conf: Sequence[float],
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


def _connected_components(adj_int: List[List[int]]) -> List[List[int]]:
    n = len(adj_int)
    seen = [False] * n
    comps: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if v != u and adj_int[u][v] > 0 and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def _match_component_to_species(comp_atoms: List[str], comp_adj: List[List[int]]) -> Optional[str]:
    n = len(comp_atoms)
    atom_count = Counter(comp_atoms)
    for name, mol in MOLECULES.items():
        if len(mol.atom_types) != n:
            continue
        if Counter(mol.atom_types) != atom_count:
            continue
        for perm in itertools.permutations(range(n)):
            ok_atoms = all(mol.atom_types[i] == comp_atoms[perm[i]] for i in range(n))
            if not ok_atoms:
                continue
            ok_bonds = True
            for i in range(n):
                for j in range(n):
                    if mol.adjacency[i][j] != comp_adj[perm[i]][perm[j]]:
                        ok_bonds = False
                        break
                if not ok_bonds:
                    break
            if ok_bonds:
                return name
    return None


def _adjacency_to_species(atom_types: List[str], pred_mat: torch.Tensor) -> Tuple[List[str], bool]:
    n = len(atom_types)
    adj_int = [[int(round(float(pred_mat[i, j].item()))) for j in range(n)] for i in range(n)]
    comps = _connected_components(adj_int)
    products: List[str] = []
    for comp in comps:
        comp_atoms = [atom_types[i] for i in comp]
        comp_adj = [[adj_int[i][j] for j in comp] for i in comp]
        name = _match_component_to_species(comp_atoms, comp_adj)
        if name is None:
            return [], False
        products.append(name)
    return products, True


def _save_unknown_species_artifacts(
    unknown_name: str,
    atom_types: Sequence[str],
    adjacency: Sequence[Sequence[int]],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    graph_path = os.path.join(out_dir, f"{unknown_name}.png")
    atoms_path = os.path.join(out_dir, f"{unknown_name}_atoms.txt")
    adj_path = os.path.join(out_dir, f"{unknown_name}_adj.txt")

    adj_t = torch.tensor(adjacency, dtype=torch.float32)
    plot_graph(adj_t, list(atom_types), graph_path, f"{unknown_name}")

    with open(atoms_path, "w", encoding="utf-8") as f:
        for i, a in enumerate(atom_types):
            f.write(f"{i}\t{a}\n")

    with open(adj_path, "w", encoding="utf-8") as f:
        for row in adjacency:
            f.write(" ".join(str(int(v)) for v in row) + "\n")


def _adjacency_to_species_or_register_unknown(
    atom_types: List[str],
    pred_mat: torch.Tensor,
    enforce_known_mapping: bool,
    unknown_cache: Dict[Tuple[Tuple[str, ...], Tuple[Tuple[int, ...], ...]], str],
    unknown_counter: List[int],
    unknown_species_dir: Optional[str] = None,
) -> Tuple[List[str], bool]:
    """
    Map predicted components to species names.
    If strict mapping is disabled, unknown components are registered as UNK_####.
    """
    n = len(atom_types)
    adj_int = [[int(round(float(pred_mat[i, j].item()))) for j in range(n)] for i in range(n)]
    comps = _connected_components(adj_int)
    products: List[str] = []

    for comp in comps:
        comp_atoms = [atom_types[i] for i in comp]
        comp_adj = [[adj_int[i][j] for j in comp] for i in comp]
        name = _match_component_to_species(comp_atoms, comp_adj)

        if name is None:
            if enforce_known_mapping:
                return [], False

            key = (
                tuple(comp_atoms),
                tuple(tuple(int(v) for v in row) for row in comp_adj),
            )
            if key in unknown_cache:
                name = unknown_cache[key]
            else:
                unknown_counter[0] += 1
                name = f"UNK_{unknown_counter[0]:04d}"
                unknown_cache[key] = name
                # Keep component adjacency separate; do not overwrite full-graph adj_int.
                comp_adj_int = [[int(v) for v in row] for row in comp_adj]
                MOLECULES[name] = MoleculeGraph(
                    atom_types=list(comp_atoms),
                    adjacency=comp_adj_int,
                )
                if unknown_species_dir:
                    _save_unknown_species_artifacts(
                        unknown_name=name,
                        atom_types=comp_atoms,
                        adjacency=comp_adj_int,
                        out_dir=unknown_species_dir,
                    )

        products.append(name)

    return products, True


def _pool_to_list(pool: Dict[str, int]) -> List[str]:
    seq: List[str] = []
    for name, count in pool.items():
        if count < 0:
            raise ValueError(f"Negative pool count for {name}: {count}")
        seq.extend([name] * count)
    return seq


def _pool_to_sorted_list(pool: Dict[str, int]) -> List[str]:
    seq: List[str] = []
    for name in sorted(pool.keys()):
        count = int(pool[name])
        if count < 0:
            raise ValueError(f"Negative pool count for {name}: {count}")
        seq.extend([name] * count)
    return seq


def _list_to_pool(items: Sequence[str]) -> Dict[str, int]:
    c = Counter(items)
    return {k: int(v) for k, v in sorted(c.items())}


def _choose_collision_group(pool_list: List[str]) -> Tuple[List[int], List[str]]:
    if not pool_list:
        return [], []
    if len(pool_list) == 1:
        idx = [0]
    else:
        k = 2 if random.random() < P_BIMOLECULAR else 1
        k = min(k, len(pool_list))
        idx = sorted(random.sample(range(len(pool_list)), k=k))
    return idx, [pool_list[i] for i in idx]


def _sample_third_body_flag(pressure: float, pressure_ref: float) -> float:
    p = pressure / (pressure + max(pressure_ref, 1e-12))
    p = max(0.0, min(1.0, p))
    return 1.0 if random.random() < p else 0.0


def _save_overall_pool_step(out_dir: str, step: int, pool: Dict[str, int]) -> None:
    steps_dir = os.path.join(out_dir, "overall_steps")
    os.makedirs(steps_dir, exist_ok=True)

    pool_path = os.path.join(steps_dir, f"pool_step_{step:03d}.txt")
    with open(pool_path, "w", encoding="utf-8") as f:
        for name, count in sorted(pool.items()):
            f.write(f"{name} {count}\n")

    mols = _pool_to_sorted_list(pool)
    graph_path = os.path.join(steps_dir, f"graph_overall_step_{step:03d}.png")
    adj_path = os.path.join(steps_dir, f"adj_overall_step_{step:03d}.txt")

    if not mols:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Empty pool", ha="center", va="center", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(graph_path, dpi=220)
        plt.close()
        with open(adj_path, "w", encoding="utf-8") as f:
            f.write("EMPTY\n")
        return

    adj, atoms = combine_species(mols)
    adj_t = torch.tensor(adj, dtype=torch.float32)
    plot_graph(adj_t, atoms, graph_path, f"Overall Pool Step {step:03d}")
    with open(adj_path, "w", encoding="utf-8") as f:
        for row in adj:
            f.write(" ".join(str(int(v)) for v in row) + "\n")


def _save_overall_gif(out_dir: str, gif_name: str, duration_ms: int) -> Optional[str]:
    steps_dir = os.path.join(out_dir, "overall_steps")
    if not os.path.isdir(steps_dir):
        return None

    frames = sorted(
        fn for fn in os.listdir(steps_dir)
        if fn.startswith("graph_overall_step_") and fn.endswith(".png")
    )
    if not frames:
        return None

    images: List[Image.Image] = []
    for fn in frames:
        path = os.path.join(steps_dir, fn)
        with Image.open(path) as im:
            images.append(im.convert("RGB"))

    if not images:
        return None

    gif_path = os.path.join(out_dir, gif_name)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, int(duration_ms)),
        loop=0,
    )
    return gif_path


def _should_save_step(step: int, total_steps: int, every: int, always_final: bool = True) -> bool:
    if every <= 1:
        return True
    if step % every == 0:
        return True
    if always_final and step == total_steps:
        return True
    return False


def _save_species_counts_plot(out_dir: str, pool_history: List[Dict[str, int]]) -> Optional[str]:
    if not pool_history:
        return None

    all_species = sorted({sp for pool in pool_history for sp in pool.keys()})
    if not all_species:
        return None

    steps = list(range(len(pool_history)))
    plt.figure(figsize=(8, 5))
    for sp in all_species:
        ys = [int(pool.get(sp, 0)) for pool in pool_history]
        if max(ys) == 0:
            continue
        plt.plot(steps, ys, marker="o", linewidth=1.8, markersize=3.5, label=sp)

    plt.xlabel("Step")
    plt.ylabel("Count")
    plt.title("Species Count vs Step")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "species_counts.png")
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def main() -> None:
    _set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if "state_dict" not in checkpoint or "model_cfg" not in checkpoint:
        raise KeyError("Checkpoint must contain 'state_dict' and 'model_cfg'.")

    model_cfg = _load_model_cfg(checkpoint["model_cfg"])
    model = EdgePredictor(model_cfg)
    model.load_state_dict(checkpoint["state_dict"])
    use_latent_branching = bool(getattr(model_cfg, "use_latent_branching", False))

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)
    model.to(device)
    model.eval()

    predict_bond_change = bool(checkpoint.get("predict_bond_change", True))
    delta_offset = 3 if predict_bond_change else 0
    index_scale = 1.0
    if model_cfg.node_in != 4:
        raise RuntimeError(f"Expected node_in=4 feature design, found node_in={model_cfg.node_in}")

    template_map = _build_branch_template_map()
    reaction_by_id = _build_reaction_map_by_id()

    rate_mean = checkpoint.get("rate_mean", None)
    rate_std = checkpoint.get("rate_std", None)
    if isinstance(rate_mean, torch.Tensor):
        rate_mean_t = rate_mean.detach().float().cpu().view(-1)
    elif rate_mean is not None:
        rate_mean_t = torch.tensor(rate_mean, dtype=torch.float32).view(-1)
    else:
        rate_mean_t = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)
    if isinstance(rate_std, torch.Tensor):
        rate_std_t = rate_std.detach().float().cpu().view(-1)
    elif rate_std is not None:
        rate_std_t = torch.tensor(rate_std, dtype=torch.float32).view(-1)
    else:
        rate_std_t = torch.ones((RATE_TARGET_DIM,), dtype=torch.float32)
    if rate_mean_t.numel() < RATE_TARGET_DIM:
        rate_mean_t = torch.cat([rate_mean_t, torch.zeros((RATE_TARGET_DIM - rate_mean_t.numel(),), dtype=torch.float32)], dim=0)
    if rate_std_t.numel() < RATE_TARGET_DIM:
        rate_std_t = torch.cat([rate_std_t, torch.ones((RATE_TARGET_DIM - rate_std_t.numel(),), dtype=torch.float32)], dim=0)
    rate_mean_t = rate_mean_t[:RATE_TARGET_DIM]
    rate_std_t = torch.clamp(rate_std_t[:RATE_TARGET_DIM], min=1e-6)

    pool = dict(INITIAL_POOL)
    pool_history: List[Dict[str, int]] = [dict(pool)]
    events_log: List[str] = []

    events_log.append(f"model: {MODEL_PATH}")
    events_log.append(f"device: {device}")
    events_log.append(f"predict_bond_change: {predict_bond_change}")
    events_log.append(f"use_latent_branching: {use_latent_branching}")
    if use_latent_branching:
        events_log.append(f"latent_samples_per_event: {LATENT_SAMPLES_PER_EVENT}")
        events_log.append(f"latent_sample_selection: {LATENT_SAMPLE_SELECTION}")
    events_log.append(f"pressure: {PRESSURE}")
    events_log.append(f"allow_unseen_reactant_channels: {ALLOW_UNSEEN_REACTANT_CHANNELS}")
    events_log.append(f"channel_selection_mode: {CHANNEL_SELECTION_MODE}")
    events_log.append(f"temperature: {TEMPERATURE}")
    events_log.append(f"enforce_known_species_mapping: {ENFORCE_KNOWN_SPECIES_MAPPING}")
    events_log.append(f"no_bond_change_log_penalty: {NO_BOND_CHANGE_LOG_PENALTY}")
    events_log.append(f"save_event_artifacts_every: {SAVE_EVENT_ARTIFACTS_EVERY}")
    events_log.append(f"save_overall_plots_every: {SAVE_OVERALL_PLOTS_EVERY}")
    events_log.append(f"save_event_local_graphs: {SAVE_EVENT_LOCAL_GRAPHS}")
    events_log.append(f"print_progress_every: {PRINT_PROGRESS_EVERY}")
    events_log.append(f"initial_pool: {pool}")
    events_log.append("")
    _save_overall_pool_step(OUT_DIR, 0, pool)

    accepted = 0
    rejected = 0
    unknown_cache: Dict[Tuple[Tuple[str, ...], Tuple[Tuple[int, ...], ...]], str] = {}
    unknown_counter = [0]
    unknown_species_dir = os.path.join(OUT_DIR, UNKNOWN_SPECIES_DIR_NAME)
    start_time = time.time()

    for step in range(1, NUM_EVENTS + 1):
        if sum(int(v) for v in pool.values()) == 0:
            events_log.append(f"step {step:03d}: STOP (empty pool)")
            if _should_save_step(step, NUM_EVENTS, SAVE_OVERALL_PLOTS_EVERY, ALWAYS_SAVE_FINAL_STEP):
                _save_overall_pool_step(OUT_DIR, step, pool)
            pool_history.append(dict(pool))
            elapsed = time.time() - start_time
            print(f"[stoch] step {step:03d}/{NUM_EVENTS}: STOP empty pool elapsed={elapsed:.1f}s")
            break

        channels: List[Dict[str, Any]] = []
        p_third = max(0.0, min(1.0, float(PRESSURE) / (float(PRESSURE) + max(float(PRESSURE_REF), EPS))))
        reactant_groups = _enumerate_collision_groups(pool)

        for reactants in reactant_groups:
            reactants_sorted = sorted(reactants)
            signature = reactant_signature(reactants_sorted)
            count_factor = _reactant_count_factor(reactants_sorted, pool)
            if count_factor <= 0.0:
                continue
            collision_prior = P_BIMOLECULAR if len(reactants_sorted) == 2 else (1.0 - P_BIMOLECULAR)
            if collision_prior <= 0.0:
                continue

            for third_body in (False, True):
                variants = template_map.get((signature, third_body), [])
                if not variants and ALLOW_UNSEEN_REACTANT_CHANNELS:
                    variants = [None]
                if not variants:
                    continue
                third_prior = p_third if third_body else (1.0 - p_third)
                if third_prior <= 0.0:
                    continue

                if use_latent_branching:
                    sample_count = max(1, int(LATENT_SAMPLES_PER_EVENT))
                    latent_channels: List[Dict[str, Any]] = []
                    for _ in range(sample_count):
                        data, atom_types = _build_inference_data(
                            reactants=reactants_sorted,
                            branch_id=0,
                            branch_count=1,
                            third_body_flag=1.0 if third_body else 0.0,
                            index_scale=index_scale,
                        )
                        data_dev = data.to(device)
                        with torch.no_grad():
                            out = model(data_dev, return_aux=True, sample_latent=True)
                            if isinstance(out, tuple):
                                logits, aux = out
                            else:
                                logits, aux = out, {}
                            pred_edges = logits.argmax(dim=-1).cpu()
                            edge_conf = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
                            conf_score = float(torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item())

                        data_cpu = data_dev.cpu()
                        pred_mat = _reconstruct_pred_matrix(
                            data=data_cpu,
                            pred_edges=pred_edges,
                            predict_bond_change=predict_bond_change,
                            delta_offset=delta_offset,
                        )
                        if APPLY_VALENCE_CAP:
                            pred_mat = _apply_valence_cap(
                                pred_mat=pred_mat,
                                atom_types=atom_types,
                                pair_i=data_cpu.pair_i.tolist(),
                                pair_j=data_cpu.pair_j.tolist(),
                                edge_conf=edge_conf,
                                valence_caps=VALENCE_CAPS,
                            )

                        known_products, known_ok = _adjacency_to_species(atom_types, pred_mat)
                        if ENFORCE_KNOWN_SPECIES_MAPPING and not known_ok:
                            continue

                        rid_hint = int(variants[0]) if len(variants) == 1 and variants[0] is not None else None
                        rxn_hint = reaction_by_id.get(rid_hint) if rid_hint is not None else None
                        rate_vec = _rate_vector_from_aux(aux, rate_mean_t, rate_std_t)
                        if rate_vec is None and rxn_hint is not None:
                            rate_vec = _rate_vector_from_template(rxn_hint)
                        if rate_vec is None:
                            rate_vec = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)

                        rate_type = _reaction_rate_type_hint(rxn_hint, third_body)
                        log_k = _log_k_effective(
                            encoded_rate_vec=rate_vec,
                            rate_type=rate_type,
                            temperature=TEMPERATURE,
                            pressure=PRESSURE,
                            pressure_ref=PRESSURE_REF,
                        )
                        if not math.isfinite(log_k):
                            continue

                        no_bond_change, changed_edges = _no_bond_change_stats(
                            react_adj=data_cpu.react_adj,
                            pred_mat=pred_mat,
                        )
                        no_change_penalty = float(NO_BOND_CHANGE_LOG_PENALTY) if no_bond_change else 0.0
                        log_prop = (
                            log_k
                            + math.log(max(count_factor, EPS))
                            + math.log(max(collision_prior, EPS))
                            + math.log(max(third_prior, EPS))
                            - no_change_penalty
                        )
                        latent_channels.append(
                            {
                                "reactants": reactants_sorted,
                                "third_body": third_body,
                                "branch_id": 0,
                                "branch_count": 1,
                                "template_reaction_id": rid_hint,
                                "data_cpu": data_cpu,
                                "atom_types": atom_types,
                                "pred_mat": pred_mat,
                                "known_products": known_products,
                                "known_ok": known_ok,
                                "rate_vec": rate_vec,
                                "rate_type": rate_type,
                                "confidence": conf_score,
                                "log_k": log_k,
                                "log_prop": log_prop,
                                "no_bond_change": no_bond_change,
                                "changed_edges": changed_edges,
                                "no_change_penalty": no_change_penalty,
                            }
                        )

                    if LATENT_SAMPLE_SELECTION == "best_confidence":
                        if latent_channels:
                            best = max(latent_channels, key=lambda c: c["confidence"])
                            channels.append(best)
                    else:
                        channels.extend(latent_channels)
                else:
                    branch_count = len(variants)
                    for branch_id, rid in enumerate(variants):
                        data, atom_types = _build_inference_data(
                            reactants=reactants_sorted,
                            branch_id=branch_id,
                            branch_count=branch_count,
                            third_body_flag=1.0 if third_body else 0.0,
                            index_scale=index_scale,
                        )
                        data_dev = data.to(device)
                        with torch.no_grad():
                            out = model(data_dev, return_aux=True, sample_latent=False)
                            if isinstance(out, tuple):
                                logits, aux = out
                            else:
                                logits, aux = out, {}
                            pred_edges = logits.argmax(dim=-1).cpu()
                            edge_conf = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
                            conf_score = float(torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item())

                        data_cpu = data_dev.cpu()
                        pred_mat = _reconstruct_pred_matrix(
                            data=data_cpu,
                            pred_edges=pred_edges,
                            predict_bond_change=predict_bond_change,
                            delta_offset=delta_offset,
                        )
                        if APPLY_VALENCE_CAP:
                            pred_mat = _apply_valence_cap(
                                pred_mat=pred_mat,
                                atom_types=atom_types,
                                pair_i=data_cpu.pair_i.tolist(),
                                pair_j=data_cpu.pair_j.tolist(),
                                edge_conf=edge_conf,
                                valence_caps=VALENCE_CAPS,
                            )

                        known_products, known_ok = _adjacency_to_species(atom_types, pred_mat)
                        if ENFORCE_KNOWN_SPECIES_MAPPING and not known_ok:
                            continue

                        rxn_hint = reaction_by_id.get(int(rid)) if rid is not None else None
                        rate_vec = _rate_vector_from_aux(aux, rate_mean_t, rate_std_t)
                        if rate_vec is None and rxn_hint is not None:
                            rate_vec = _rate_vector_from_template(rxn_hint)
                        if rate_vec is None:
                            rate_vec = torch.zeros((RATE_TARGET_DIM,), dtype=torch.float32)

                        rate_type = _reaction_rate_type_hint(rxn_hint, third_body)
                        log_k = _log_k_effective(
                            encoded_rate_vec=rate_vec,
                            rate_type=rate_type,
                            temperature=TEMPERATURE,
                            pressure=PRESSURE,
                            pressure_ref=PRESSURE_REF,
                        )
                        if not math.isfinite(log_k):
                            continue

                        no_bond_change, changed_edges = _no_bond_change_stats(
                            react_adj=data_cpu.react_adj,
                            pred_mat=pred_mat,
                        )
                        no_change_penalty = float(NO_BOND_CHANGE_LOG_PENALTY) if no_bond_change else 0.0
                        log_prop = (
                            log_k
                            + math.log(max(count_factor, EPS))
                            + math.log(max(collision_prior, EPS))
                            + math.log(max(third_prior, EPS))
                            - no_change_penalty
                        )
                        channels.append(
                            {
                                "reactants": reactants_sorted,
                                "third_body": third_body,
                                "branch_id": branch_id,
                                "branch_count": branch_count,
                                "template_reaction_id": (int(rid) if rid is not None else -1),
                                "data_cpu": data_cpu,
                                "atom_types": atom_types,
                                "pred_mat": pred_mat,
                                "known_products": known_products,
                                "known_ok": known_ok,
                                "rate_vec": rate_vec,
                                "rate_type": rate_type,
                                "confidence": conf_score,
                                "log_k": log_k,
                                "log_prop": log_prop,
                                "no_bond_change": no_bond_change,
                                "changed_edges": changed_edges,
                                "no_change_penalty": no_change_penalty,
                            }
                        )

        if not channels:
            rejected += 1
            events_log.append(f"step {step:03d}: REJECT (no valid channels) pool={pool}")
            if _should_save_step(step, NUM_EVENTS, SAVE_OVERALL_PLOTS_EVERY, ALWAYS_SAVE_FINAL_STEP):
                _save_overall_pool_step(OUT_DIR, step, pool)
            pool_history.append(dict(pool))
            continue

        max_log_prop = max(float(c["log_prop"]) for c in channels)
        weights = [math.exp(float(c["log_prop"]) - max_log_prop) for c in channels]
        sum_w = sum(weights)
        if sum_w <= 0.0:
            probs = [1.0 / len(channels)] * len(channels)
        else:
            probs = [w / sum_w for w in weights]

        mode = str(CHANNEL_SELECTION_MODE).strip().lower()
        if mode == "argmax":
            max_p = max(probs)
            winners = [i for i, p in enumerate(probs) if abs(p - max_p) <= 1e-15]
            pick = random.choice(winners)
        elif mode == "sample":
            u = random.random()
            cum = 0.0
            pick = len(channels) - 1
            for i, p in enumerate(probs):
                cum += p
                if u <= cum:
                    pick = i
                    break
        else:
            raise ValueError(f"Unknown CHANNEL_SELECTION_MODE={CHANNEL_SELECTION_MODE!r}. Use 'argmax' or 'sample'.")

        chosen = channels[pick]
        reactants = list(chosen["reactants"])
        third_body = bool(chosen["third_body"])
        branch_id = int(chosen["branch_id"])
        branch_count = int(chosen["branch_count"])
        data_cpu = chosen["data_cpu"]
        atom_types = list(chosen["atom_types"])
        pred_mat = chosen["pred_mat"]

        if bool(chosen["known_ok"]):
            products = list(chosen["known_products"])
            ok = True
        else:
            products, ok = _adjacency_to_species_or_register_unknown(
                atom_types=atom_types,
                pred_mat=pred_mat,
                enforce_known_mapping=ENFORCE_KNOWN_SPECIES_MAPPING,
                unknown_cache=unknown_cache,
                unknown_counter=unknown_counter,
                unknown_species_dir=unknown_species_dir,
            )

        if _should_save_step(step, NUM_EVENTS, SAVE_EVENT_ARTIFACTS_EVERY, ALWAYS_SAVE_FINAL_STEP):
            step_dir = os.path.join(OUT_DIR, f"event_{step:03d}")
            os.makedirs(step_dir, exist_ok=True)
            if SAVE_EVENT_LOCAL_GRAPHS:
                plot_graph(
                    data_cpu.react_adj,
                    atom_types,
                    os.path.join(step_dir, "graph_reactants.png"),
                    f"Reactants {step}",
                )
                plot_graph(
                    pred_mat,
                    atom_types,
                    os.path.join(step_dir, "graph_predicted.png"),
                    f"Predicted {step}",
                )
            with open(os.path.join(step_dir, "reactants.txt"), "w", encoding="utf-8") as f:
                f.write(" ".join(reactants) + "\n")
            with open(os.path.join(step_dir, "predicted_products.txt"), "w", encoding="utf-8") as f:
                f.write((" ".join(products) if ok else "UNKNOWN") + "\n")
            with open(os.path.join(step_dir, "selected_channel.txt"), "w", encoding="utf-8") as f:
                f.write(f"candidate_channels {len(channels)}\n")
                f.write(f"selected_index {pick}\n")
                f.write(f"selected_probability {probs[pick]:.8f}\n")
                f.write(f"log_k {float(chosen['log_k']):.8f}\n")
                f.write(f"no_bond_change {int(bool(chosen.get('no_bond_change', False)))}\n")
                f.write(f"changed_edges {int(chosen.get('changed_edges', -1))}\n")
                f.write(f"no_change_penalty {float(chosen.get('no_change_penalty', 0.0)):.6f}\n")
                f.write(f"rate_type {chosen['rate_type']}\n")
                f.write(f"template_reaction_id {chosen['template_reaction_id']}\n")
            with open(os.path.join(step_dir, "pred_mat.txt"), "w", encoding="utf-8") as f:
                for row in pred_mat.tolist():
                    f.write(" ".join(f"{v:.1f}" for v in row) + "\n")

        if ok:
            new_pool = dict(pool)
            feasible = True
            for r in reactants:
                c = int(new_pool.get(r, 0))
                if c <= 0:
                    feasible = False
                    break
                new_pool[r] = c - 1
                if new_pool[r] == 0:
                    del new_pool[r]
            if feasible:
                accepted += 1
                for p in products:
                    new_pool[p] = int(new_pool.get(p, 0)) + 1
                pool = {k: int(v) for k, v in sorted(new_pool.items()) if int(v) > 0}
                events_log.append(
                    f"step {step:03d}: ACCEPT reactants={reactants} third_body={int(third_body)} "
                    f"branch={branch_id}/{branch_count-1} p={probs[pick]:.4f} "
                    f"channels={len(channels)} -> products={products} pool={pool}"
                )
            else:
                rejected += 1
                pool = dict(pool)
                events_log.append(
                    f"step {step:03d}: REJECT reactants={reactants} third_body={int(third_body)} "
                    f"branch={branch_id}/{branch_count-1} p={probs[pick]:.4f} "
                    f"channels={len(channels)} -> insufficient reactants pool={pool}"
                )
        else:
            rejected += 1
            # Keep pool unchanged on rejected events.
            pool = dict(pool)
            events_log.append(
                f"step {step:03d}: REJECT reactants={reactants} third_body={int(third_body)} "
                f"branch={branch_id}/{branch_count-1} p={probs[pick]:.4f} "
                f"channels={len(channels)} -> products=UNKNOWN pool={pool}"
            )

        if _should_save_step(step, NUM_EVENTS, SAVE_OVERALL_PLOTS_EVERY, ALWAYS_SAVE_FINAL_STEP):
            _save_overall_pool_step(OUT_DIR, step, pool)
        pool_history.append(dict(pool))

        if PRINT_PROGRESS_EVERY > 0 and (
            step == 1 or step == NUM_EVENTS or step % PRINT_PROGRESS_EVERY == 0
        ):
            elapsed = time.time() - start_time
            steps_done = max(1, step)
            step_rate = steps_done / max(elapsed, 1e-9)
            eta = (NUM_EVENTS - step) / max(step_rate, 1e-9)
            pool_total = sum(int(v) for v in pool.values())
            pool_species = sum(1 for v in pool.values() if int(v) > 0)
            print(
                f"[stoch] step {step:03d}/{NUM_EVENTS} "
                f"accepted={accepted} rejected={rejected} "
                f"pool_total={pool_total} species={pool_species} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )

    events_log.append("")
    events_log.append(f"accepted_events: {accepted}")
    events_log.append(f"rejected_events: {rejected}")
    if not ENFORCE_KNOWN_SPECIES_MAPPING:
        events_log.append(f"registered_unknown_species: {unknown_counter[0]}")
    events_log.append(f"final_pool: {pool}")

    with open(os.path.join(OUT_DIR, "events.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(events_log) + "\n")

    gif_path = None
    if SAVE_OVERALL_GIF:
        gif_path = _save_overall_gif(
            out_dir=OUT_DIR,
            gif_name=OVERALL_GIF_NAME,
            duration_ms=OVERALL_GIF_DURATION_MS,
        )
    species_plot_path = _save_species_counts_plot(OUT_DIR, pool_history)

    print(f"Saved stochastic inference outputs to: {OUT_DIR}")
    if gif_path:
        print(f"Saved overall-step GIF: {gif_path}")
    if species_plot_path:
        print(f"Saved species-count plot: {species_plot_path}")
    print(f"Accepted: {accepted}, Rejected: {rejected}")
    print(f"Final pool: {pool}")


if __name__ == "__main__":
    main()
