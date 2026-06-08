import json
import re
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional


@dataclass(frozen=True)
class MoleculeGraph:
    atom_types: List[str]
    adjacency: List[List[int]]


# Bond order: 0 = no bond, 1 = single, 2 = double.
MOLECULES: Dict[str, MoleculeGraph] = {
    "H": MoleculeGraph(atom_types=["H"], adjacency=[[0]]),
    "O": MoleculeGraph(atom_types=["O"], adjacency=[[0]]),
    "H2": MoleculeGraph(
        atom_types=["H", "H"],
        adjacency=[
            [0, 1],
            [1, 0],
        ],
    ),
    "O2": MoleculeGraph(
        atom_types=["O", "O"],
        adjacency=[
            [0, 2],
            [2, 0],
        ],
    ),
    "OH": MoleculeGraph(
        atom_types=["O", "H"],
        adjacency=[
            [0, 1],
            [1, 0],
        ],
    ),
    "HO2": MoleculeGraph(
        atom_types=["O", "O", "H"],  # O1-O2 single bond, H on O1
        adjacency=[
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
    ),
    "H2O": MoleculeGraph(
        atom_types=["O", "H", "H"],
        adjacency=[
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
    ),
    "H2O2": MoleculeGraph(
        atom_types=["O", "O", "H", "H"],
        adjacency=[
            [0, 1, 1, 0],  # H-O-O-H
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
    ),
}


RATE_UNITS = {
    "A": "cm, mol, s (CTI convention)",
    "Ea": "cal/mol",
    "temperature": "K",
}


REACTIONS = [
    {
        "id": 1,
        "equation": "H + O2 <=> OH + O",
        "rate": {"type": "arrhenius", "A": 3.520000e16, "n": -0.7, "Ea": 17069.79},
    },
    {
        "id": 2,
        "equation": "H2 + O <=> OH + H",
        "rate": {"type": "arrhenius", "A": 5.060000e4, "n": 2.67, "Ea": 6290.63},
    },
    {
        "id": 3,
        "equation": "H2 + OH <=> H2O + H",
        "rate": {"type": "arrhenius", "A": 1.170000e9, "n": 1.3, "Ea": 3635.28},
    },
    {
        "id": 4,
        "equation": "H2O + O <=> 2 OH",
        "rate": {"type": "arrhenius", "A": 7.600000e0, "n": 3.84, "Ea": 12779.64},
    },
    {
        "id": 5,
        "equation": "2 H + M <=> H2 + M",
        "rate": {
            "type": "three_body",
            "A": 1.300000e18,
            "n": -1.0,
            "Ea": 0.0,
            "efficiencies": {"H2": 2.5, "H2O": 12.0},
        },
    },
    {
        "id": 6,
        "equation": "H + OH + M <=> H2O + M",
        "rate": {
            "type": "three_body",
            "A": 4.000000e22,
            "n": -2.0,
            "Ea": 0.0,
            "efficiencies": {"H2": 2.5, "H2O": 12.0},
        },
    },
    {
        "id": 7,
        "equation": "2 O + M <=> O2 + M",
        "rate": {
            "type": "three_body",
            "A": 6.170000e15,
            "n": -0.5,
            "Ea": 0.0,
            "efficiencies": {"H2": 2.5, "H2O": 12.0},
        },
    },
    {
        "id": 8,
        "equation": "H + O + M <=> OH + M",
        "rate": {
            "type": "three_body",
            "A": 4.710000e18,
            "n": -1.0,
            "Ea": 0.0,
            "efficiencies": {"H2": 2.5, "H2O": 12.0},
        },
    },
    {
        "id": 9,
        "equation": "O + OH + M <=> HO2 + M",
        "rate": {
            "type": "three_body",
            "A": 8.000000e15,
            "n": 0.0,
            "Ea": 0.0,
            "efficiencies": {"H2": 2.5, "H2O": 12.0},
        },
    },
    {
        "id": 10,
        "equation": "H + O2 (+ M) <=> HO2 (+ M)",
        "rate": {
            "type": "falloff",
            "high": {"A": 4.650000e12, "n": 0.44, "Ea": 0.0},
            "low": {"A": 5.750000e19, "n": -1.4, "Ea": 0.0},
            "efficiencies": {"H2": 2.5, "H2O": 16.0},
            "falloff": {"model": "Troe", "A": 0.5, "T3": 1.0e-30, "T1": 1.0e30},
        },
    },
    {
        "id": 11,
        "equation": "HO2 + H <=> 2 OH",
        "rate": {"type": "arrhenius", "A": 7.080000e13, "n": 0.0, "Ea": 295.0},
    },
    {
        "id": 12,
        "equation": "HO2 + H <=> H2 + O2",
        "rate": {"type": "arrhenius", "A": 1.660000e13, "n": 0.0, "Ea": 822.9},
    },
    {
        "id": 13,
        "equation": "HO2 + H <=> H2O + O",
        "rate": {"type": "arrhenius", "A": 3.100000e13, "n": 0.0, "Ea": 1720.84},
    },
    {
        "id": 14,
        "equation": "HO2 + O <=> OH + O2",
        "rate": {"type": "arrhenius", "A": 2.000000e13, "n": 0.0, "Ea": 0.0},
    },
    {
        "id": 15,
        "equation": "HO2 + OH <=> H2O + O2",
        "rate": {"type": "arrhenius", "A": 2.890000e13, "n": 0.0, "Ea": -497.13},
    },
    {
        "id": 16,
        "equation": "2 OH (+ M) <=> H2O2 (+ M)",
        "rate": {
            "type": "falloff",
            "high": {"A": 7.400000e13, "n": -0.37, "Ea": 0.0},
            "low": {"A": 2.300000e18, "n": -0.9, "Ea": -1701.72},
            "efficiencies": {"H2": 2.0, "H2O": 6.0},
            "falloff": {"model": "Troe", "A": 0.735, "T3": 94.0, "T1": 1756.0, "T2": 5182.0},
        },
    },
    {
        "id": 17,
        "equation": "2 HO2 <=> H2O2 + O2",
        "rate": {"type": "arrhenius", "A": 3.020000e12, "n": 0.0, "Ea": 1386.23},
    },
    {
        "id": 18,
        "equation": "H2O2 + H <=> HO2 + H2",
        "rate": {"type": "arrhenius", "A": 2.300000e13, "n": 0.0, "Ea": 7950.05},
    },
    {
        "id": 19,
        "equation": "H2O2 + H <=> H2O + OH",
        "rate": {"type": "arrhenius", "A": 1.000000e13, "n": 0.0, "Ea": 3585.09},
    },
    {
        "id": 20,
        "equation": "H2O2 + OH <=> H2O + HO2",
        "rate": {"type": "arrhenius", "A": 7.080000e12, "n": 0.0, "Ea": 1434.03},
    },
    {
        "id": 21,
        "equation": "H2O2 + O <=> HO2 + OH",
        "rate": {"type": "arrhenius", "A": 9.630000e6, "n": 2.0, "Ea": 3991.4},
    },
]


def parse_equation(equation: str) -> Tuple[List[str], List[str]]:
    """Split a reaction string into expanded reactant/product species lists."""
    cleaned = re.sub(r"\(\s*\+\s*M\s*\)", "", equation)  # drop explicit (+ M)
    cleaned = re.sub(r"\bM\b", "", cleaned)  # drop bare third bodies
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    reactant_str, product_str = [side.strip() for side in cleaned.split("<=>")]
    return _expand_side(reactant_str), _expand_side(product_str)


def _expand_side(side: str) -> List[str]:
    """Return species list with stoichiometric coefficients expanded."""
    species: List[str] = []
    if not side:
        return species
    for term in side.split("+"):
        token = term.strip()
        if not token or token == "M":
            continue
        match = re.match(r"(?:(\d+)\s*)?([A-Za-z0-9]+)", token)
        if not match:
            raise ValueError(f"Cannot parse species token: '{token}'")
        count = int(match.group(1) or 1)
        name = match.group(2)
        if name == "M":  # already filtered, but keep for safety
            continue
        if name not in MOLECULES:
            raise KeyError(f"Unknown species '{name}' in token '{token}'")
        species.extend([name] * count)
    return species


def _atom_count(species_list: List[str]) -> int:
    return sum(len(MOLECULES[name].atom_types) for name in species_list)


def reactant_signature(species_list: List[str]) -> Tuple[Tuple[str, int], ...]:
    """Canonical signature for grouping reactions with identical reactants."""
    counts: Dict[str, int] = {}
    for name in species_list:
        counts[name] = counts.get(name, 0) + 1
    return tuple(sorted(counts.items()))


def _build_type_fifo_permutation(target_atoms: List[str], source_atoms: List[str]) -> List[int]:
    """Stable type-only mapping: first target atom of type X gets first source atom of type X."""
    positions: Dict[str, List[int]] = {}
    for idx, atom in enumerate(source_atoms):
        positions.setdefault(atom, []).append(idx)

    perm: List[int] = []
    for atom in target_atoms:
        if atom not in positions or not positions[atom]:
            raise ValueError(f"No available source atom to match target atom '{atom}'.")
        perm.append(positions[atom].pop(0))
    return perm


def _node_type_signature(
    idx: int,
    atoms: List[str],
    adj: List[List[int]],
    atom_order: List[str],
) -> Tuple[int, int, Tuple[int, ...]]:
    """
    Lightweight local environment signature used to seed atom mapping.
    - degree (# bonded neighbors)
    - bond-order sum
    - neighbor counts per atom type in atom_order
    """
    degree = 0
    bond_sum = 0
    nbr_count = {a: 0 for a in atom_order}
    row = adj[idx]
    for j, b in enumerate(row):
        w = int(round(float(b)))
        if w <= 0:
            continue
        degree += 1
        bond_sum += w
        nbr_count[atoms[j]] = nbr_count.get(atoms[j], 0) + 1
    return degree, bond_sum, tuple(nbr_count.get(a, 0) for a in atom_order)


def _build_signature_seed_permutation(
    target_atoms: List[str],
    source_atoms: List[str],
    target_adj: List[List[int]],
    source_adj: List[List[int]],
) -> List[int]:
    """
    Build a deterministic initial permutation by matching local atom signatures
    within each atom type.
    """
    atom_order = sorted(set(target_atoms) | set(source_atoms))
    target_by_type: Dict[str, List[int]] = {}
    source_by_type: Dict[str, List[int]] = {}
    for i, a in enumerate(target_atoms):
        target_by_type.setdefault(a, []).append(i)
    for j, a in enumerate(source_atoms):
        source_by_type.setdefault(a, []).append(j)

    perm = [-1] * len(target_atoms)
    for atom in sorted(target_by_type.keys()):
        t_idx = target_by_type[atom]
        s_idx = source_by_type.get(atom, [])
        if len(t_idx) != len(s_idx):
            raise ValueError(
                f"Atom multiplicity mismatch for '{atom}': target {len(t_idx)} vs source {len(s_idx)}"
            )
        t_sorted = sorted(
            t_idx,
            key=lambda i: (_node_type_signature(i, target_atoms, target_adj, atom_order), i),
        )
        s_sorted = sorted(
            s_idx,
            key=lambda i: (_node_type_signature(i, source_atoms, source_adj, atom_order), i),
        )
        for ti, si in zip(t_sorted, s_sorted):
            perm[ti] = si

    if any(p < 0 for p in perm):
        raise RuntimeError("Failed to build complete signature-seeded permutation.")
    return perm


def _mapping_edit_objective(
    target_adj: List[List[int]],
    source_adj: List[List[int]],
    perm: List[int],
) -> int:
    """Sum of absolute bond-order edits after applying mapping."""
    n = len(perm)
    total = 0
    for i in range(n):
        pi = perm[i]
        for j in range(i + 1, n):
            pj = perm[j]
            total += abs(int(target_adj[i][j]) - int(source_adj[pi][pj]))
    return total


def _swap_delta_objective(
    target_adj: List[List[int]],
    source_adj: List[List[int]],
    perm: List[int],
    i: int,
    j: int,
) -> int:
    """
    Objective delta for swapping mapped source indices of target atoms i and j.
    Negative delta means objective improvement.
    """
    if i == j:
        return 0
    pi = perm[i]
    pj = perm[j]
    n = len(perm)
    delta = 0
    for k in range(n):
        if k == i or k == j:
            continue
        pk = perm[k]
        old_ik = abs(int(target_adj[i][k]) - int(source_adj[pi][pk]))
        new_ik = abs(int(target_adj[i][k]) - int(source_adj[pj][pk]))
        old_jk = abs(int(target_adj[j][k]) - int(source_adj[pj][pk]))
        new_jk = abs(int(target_adj[j][k]) - int(source_adj[pi][pk]))
        delta += (new_ik + new_jk) - (old_ik + old_jk)
    # Pair (i,j) term is unchanged for symmetric adjacency: source[pi][pj] == source[pj][pi].
    return delta


def _improve_permutation_by_type_swaps(
    target_atoms: List[str],
    target_adj: List[List[int]],
    source_adj: List[List[int]],
    perm: List[int],
) -> List[int]:
    """
    Hill-climb by swapping atoms of the same type to reduce bond-edit objective.
    This keeps type constraints exactly satisfied.
    """
    idx_by_type: Dict[str, List[int]] = {}
    for i, a in enumerate(target_atoms):
        idx_by_type.setdefault(a, []).append(i)

    # Safety bounds for larger molecules.
    n = len(target_atoms)
    max_passes = max(1, 2 * n)
    eval_budget = max(5000, n * n * 20)
    eval_count = 0

    for _ in range(max_passes):
        improved = False
        for atom in sorted(idx_by_type.keys()):
            idxs = idx_by_type[atom]
            m = len(idxs)
            if m < 2:
                continue
            for a in range(m - 1):
                i = idxs[a]
                for b in range(a + 1, m):
                    j = idxs[b]
                    delta = _swap_delta_objective(target_adj, source_adj, perm, i, j)
                    eval_count += 1
                    if delta < 0:
                        perm[i], perm[j] = perm[j], perm[i]
                        improved = True
                        break
                    if eval_count >= eval_budget:
                        return perm
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return perm


def reorder_to_target(
    target_atoms: List[str],
    source_atoms: List[str],
    source_adj: List[List[int]],
    target_adj: Optional[List[List[int]]] = None,
) -> List[List[int]]:
    """
    Reorder source adjacency to match target atom ordering.

    If target_adj is provided, mapping is optimized to minimize total bond-order
    edits between target_adj and reordered source_adj under atom-type constraints.
    """
    if len(target_atoms) != len(source_atoms):
        raise ValueError("Target and source atom lists differ in length.")

    # Always keep a stable FIFO fallback for robustness.
    perm_fifo = _build_type_fifo_permutation(target_atoms, source_atoms)
    perm = perm_fifo

    if target_adj is not None:
        n = len(target_atoms)
        if len(target_adj) != n or len(source_adj) != n:
            raise ValueError("Adjacency sizes do not match atom list lengths.")

        # Signature-seeded + local swap refinement.
        perm_seed = _build_signature_seed_permutation(
            target_atoms=target_atoms,
            source_atoms=source_atoms,
            target_adj=target_adj,
            source_adj=source_adj,
        )
        perm_refined = _improve_permutation_by_type_swaps(
            target_atoms=target_atoms,
            target_adj=target_adj,
            source_adj=source_adj,
            perm=perm_seed,
        )

        # Never degrade versus stable FIFO mapping.
        obj_fifo = _mapping_edit_objective(target_adj, source_adj, perm_fifo)
        obj_refined = _mapping_edit_objective(target_adj, source_adj, perm_refined)
        perm = perm_refined if obj_refined <= obj_fifo else perm_fifo

    size = len(target_atoms)
    reordered = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        si = perm[i]
        for j in range(size):
            sj = perm[j]
            reordered[i][j] = source_adj[si][sj]
    return reordered


def combine_species(species_list: List[str]) -> Tuple[List[List[int]], List[str]]:
    """Stack molecule graphs block-diagonally for a reaction side."""
    total_atoms = _atom_count(species_list)
    adjacency = [[0 for _ in range(total_atoms)] for _ in range(total_atoms)]
    atom_types: List[str] = []
    offset = 0
    for name in species_list:
        mol = MOLECULES[name]
        size = len(mol.atom_types)
        atom_types.extend(mol.atom_types)
        for i in range(size):
            for j in range(size):
                adjacency[offset + i][offset + j] = mol.adjacency[i][j]
        offset += size
    return adjacency, atom_types


def _pad_matrix(matrix: List[List[int]], target_size: int) -> List[List[int]]:
    current = len(matrix)
    if current > target_size:
        raise ValueError(f"Matrix size {current} exceeds target {target_size}")
    if current == target_size:
        return matrix
    padded = [row + [0] * (target_size - current) for row in matrix]
    for _ in range(target_size - current):
        padded.append([0] * target_size)
    return padded


def _pad_atom_types(atom_types: List[str], target_size: int, pad_token: str = "PAD") -> List[str]:
    if len(atom_types) > target_size:
        raise ValueError(f"Atom type list length {len(atom_types)} exceeds target {target_size}")
    return atom_types + [pad_token] * (target_size - len(atom_types))


def compute_max_atoms(reaction_equations: List[str]) -> int:
    max_atoms = 0
    for eq in reaction_equations:
        reactants, products = parse_equation(eq)
        max_atoms = max(max_atoms, _atom_count(reactants), _atom_count(products))
    return max_atoms


GLOBAL_MAX_ATOMS = compute_max_atoms([r["equation"] for r in REACTIONS])
# Default padded size for model inputs/outputs.
PAD_SIZE = 18


def build_reaction_graphs(target_size: int = PAD_SIZE) -> List[Dict]:
    """Return padded adjacency/atom-type pairs for each reaction."""
    entries = []
    for reaction in REACTIONS:
        reactants, products = parse_equation(reaction["equation"])
        r_adj, r_atoms = combine_species(reactants)
        p_adj, p_atoms = combine_species(products)
        if _atom_count(reactants) != _atom_count(products):
            raise ValueError(f"Atom count mismatch in reaction '{reaction['equation']}'")
        p_adj = reorder_to_target(
            target_atoms=r_atoms,
            source_atoms=p_atoms,
            source_adj=p_adj,
            target_adj=r_adj,
        )
        entries.append(
            {
                "id": reaction["id"],
                "equation": reaction["equation"],
                "reactant_atom_types": _pad_atom_types(r_atoms, target_size),
                "product_atom_types": _pad_atom_types(r_atoms, target_size),
                "reactant_adjacency": _pad_matrix(r_adj, target_size),
                "product_adjacency": _pad_matrix(p_adj, target_size),
            }
        )
    return entries


def save_as_json(path: str, target_size: int = GLOBAL_MAX_ATOMS) -> None:
    graphs = build_reaction_graphs(target_size=target_size)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(graphs, fp, indent=2)


def _repeat_block(adj: List[List[int]], atoms: List[str], copies: int) -> Tuple[List[List[int]], List[str]]:
    """Tile a block-diagonal adjacency/atom list several times."""
    block_size = len(atoms)
    total_atoms = block_size * copies
    adjacency = [[0 for _ in range(total_atoms)] for _ in range(total_atoms)]
    atom_types: List[str] = []
    for c in range(copies):
        offset = c * block_size
        atom_types.extend(atoms)
        for i in range(block_size):
            for j in range(block_size):
                adjacency[offset + i][offset + j] = adj[i][j]
    return adjacency, atom_types


def _stack_blocks(blocks: List[Tuple[List[List[int]], List[str]]]) -> Tuple[List[List[int]], List[str]]:
    """Place adjacency blocks along the diagonal in the order provided."""
    total_atoms = sum(len(atoms) for _, atoms in blocks)
    adjacency = [[0 for _ in range(total_atoms)] for _ in range(total_atoms)]
    atom_types: List[str] = []
    offset = 0
    for adj, atoms in blocks:
        size = len(atoms)
        atom_types.extend(atoms)
        for i in range(size):
            for j in range(size):
                adjacency[offset + i][offset + j] = adj[i][j]
        offset += size
    return adjacency, atom_types


def build_grouped_training_set(target_size: int = PAD_SIZE) -> List[Dict]:
    """
    Group reactions by identical reactants, duplicate the group to fill up to
    target_size atoms (without exceeding), and pad the remainder.
    """
    # Bucket reactions by reactant signature.
    groups: Dict[Tuple[Tuple[str, int], ...], List[Dict]] = {}
    for rxn in REACTIONS:
        reactants, _ = parse_equation(rxn["equation"])
        key = reactant_signature(reactants)
        groups.setdefault(key, []).append({**rxn, "reactants": reactants})

    grouped_entries = []
    for idx, (sig, rxns) in enumerate(groups.items(), start=1):
        base_reactants = rxns[0]["reactants"]
        base_adj, base_atoms = combine_species(base_reactants)
        base_size = len(base_atoms)

        # One copy of reactants per reaction in the group.
        group_adj, group_atoms = _repeat_block(base_adj, base_atoms, copies=len(rxns))
        group_atom_count = len(group_atoms)

        # Duplicate the whole group to approach target_size without exceeding.
        group_copies = max(1, target_size // group_atom_count)
        total_adj, total_atoms = _repeat_block(group_adj, group_atoms, copies=group_copies)
        total_atoms_count = len(total_atoms)
        if total_atoms_count > target_size:
            raise ValueError(
                f"Group {idx} exceeds target atoms ({total_atoms_count} > {target_size})"
            )

        entry = {
            "group_id": idx,
            "reactant_signature": sig,
            "reactant_atom_types": _pad_atom_types(total_atoms, target_size),
            "reactant_adjacency": _pad_matrix(total_adj, target_size),
            "group_atom_count": total_atoms_count,
            "group_copies": group_copies,
            "member_reactions": [],
        }

        product_blocks: List[Tuple[List[List[int]], List[str]]] = []
        for rxn in rxns:
            _, products = parse_equation(rxn["equation"])
            p_adj, p_atoms = combine_species(products)
            if len(p_atoms) != base_size:
                raise ValueError(
                    f"Atom count mismatch between reactants ({base_size}) and products ({len(p_atoms)}) "
                    f"in reaction {rxn['id']}"
                )
            product_blocks.append((p_adj, p_atoms))
            entry["member_reactions"].append(
                {
                    "id": rxn["id"],
                    "equation": rxn["equation"],
                    "product_atom_types": p_atoms,
                    "product_adjacency": p_adj,
                }
            )

        # Arrange products in the same tiled order as reactants: repeat the product set
        # group_copies times, keeping reaction order.
        tiled_product_blocks = []
        for _ in range(group_copies):
            tiled_product_blocks.extend(product_blocks)
        prod_adj, prod_atoms = _stack_blocks(tiled_product_blocks)
        prod_adj = reorder_to_target(
            target_atoms=total_atoms,
            source_atoms=prod_atoms,
            source_adj=prod_adj,
            target_adj=total_adj,
        )
        prod_atoms = total_atoms.copy()
        if len(prod_atoms) > target_size:
            raise ValueError(
                f"Tiled product atoms {len(prod_atoms)} exceed target size {target_size} for group {idx}"
            )
        entry["product_atom_types"] = _pad_atom_types(prod_atoms, target_size)
        entry["product_adjacency"] = _pad_matrix(prod_adj, target_size)

        grouped_entries.append(entry)

    return grouped_entries


def save_grouped_json(path: str, target_size: int = PAD_SIZE) -> None:
    graphs = build_grouped_training_set(target_size=target_size)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(graphs, fp, indent=2)


def build_validation_set(
    n: int = 10,
    target_size: int = PAD_SIZE,
    seed: int = 0,
    include_grouped: bool = True,
    exclude_ids: Optional[Set[int]] = None,
    pack_single: bool = False,
) -> List[Dict]:
    """
    Build validation samples. If pack_single is False, pack multiple reactions
    until the atom count would exceed target_size; otherwise each sample is a
    single reaction padded to target_size.
    """
    rng = random.Random(seed)
    exclude_ids = exclude_ids or set()
    unused = [r for r in REACTIONS if r["id"] not in exclude_ids]
    used_ids: Set[int] = set()
    entries: List[Dict] = []

    if pack_single:
        if not unused:
            raise ValueError("No reactions available for sampling.")
        sampled = rng.sample(unused, k=min(n, len(unused)))
        for rxn in sampled:
            reactants, products = parse_equation(rxn["equation"])
            r_adj, r_atoms = combine_species(reactants)
            p_adj, p_atoms = combine_species(products)
            if _atom_count(reactants) != _atom_count(products):
                raise ValueError(f"Atom count mismatch in reaction '{rxn['equation']}'")
            p_adj = reorder_to_target(
                target_atoms=r_atoms,
                source_atoms=p_atoms,
                source_adj=p_adj,
                target_adj=r_adj,
            )
            entries.append(
                {
                    "equation": rxn["equation"],
                    "member_reactions": [{"id": rxn["id"], "equation": rxn["equation"]}],
                    "reactant_atom_types": _pad_atom_types(r_atoms, target_size),
                    "product_atom_types": _pad_atom_types(p_atoms, target_size),
                    "reactant_adjacency": _pad_matrix(r_adj, target_size),
                    "product_adjacency": _pad_matrix(p_adj, target_size),
                    "atom_count": len(r_atoms),
                }
            )
        if include_grouped:
            entries.extend(build_grouped_training_set(target_size=target_size))
        return entries

    while len(entries) < n:
        if not unused:
            unused = [r for r in REACTIONS if r["id"] not in exclude_ids]
            if not unused:
                raise ValueError(
                    "No reactions available for sampling after applying exclusions; "
                    "try lowering n or relaxing exclude_ids."
                )
        rng.shuffle(unused)

        blocks: List[Tuple[List[List[int]], List[str], List[List[int]]]] = []
        chosen: List[Dict] = []
        total_atoms = 0

        for rxn in list(unused):
            reactants, products = parse_equation(rxn["equation"])
            size = _atom_count(reactants)
            if size > target_size:
                continue
            if total_atoms + size > target_size:
                continue
            if rxn["id"] not in used_ids and len(used_ids) + 1 >= len(REACTIONS):
                continue  # leave at least one reaction unused for training
            r_adj, r_atoms = combine_species(reactants)
            p_adj, p_atoms = combine_species(products)
            if _atom_count(reactants) != _atom_count(products):
                raise ValueError(f"Atom count mismatch in reaction '{rxn['equation']}'")
            p_adj = reorder_to_target(
                target_atoms=r_atoms,
                source_atoms=p_atoms,
                source_adj=p_adj,
                target_adj=r_adj,
            )
            blocks.append((r_adj, r_atoms, p_adj))
            chosen.append(rxn)
            used_ids.add(rxn["id"])
            total_atoms += size
            if total_atoms == target_size:
                break

        if not blocks:
            # Fallback: pick the smallest remaining reaction.
            if not unused:
                break
            rxn = min(unused, key=lambda r: _atom_count(parse_equation(r["equation"])[0]))
            if rxn["id"] not in used_ids and len(used_ids) + 1 >= len(REACTIONS):
                break
            reactants, products = parse_equation(rxn["equation"])
            r_adj, r_atoms = combine_species(reactants)
            p_adj, p_atoms = combine_species(products)
            if _atom_count(reactants) != _atom_count(products):
                raise ValueError(f"Atom count mismatch in reaction '{rxn['equation']}'")
            p_adj = reorder_to_target(
                target_atoms=r_atoms,
                source_atoms=p_atoms,
                source_adj=p_adj,
                target_adj=r_adj,
            )
            blocks.append((r_adj, r_atoms, p_adj))
            chosen.append(rxn)
            used_ids.add(rxn["id"])
            total_atoms = _atom_count(reactants)

        react_adj, react_atoms = _stack_blocks([(b[0], b[1]) for b in blocks])
        prod_adj, prod_atoms = _stack_blocks([(b[2], b[1]) for b in blocks])
        if len(prod_atoms) != len(react_atoms):
            raise ValueError("Product and reactant atom ordering mismatch after stacking.")

        entries.append(
            {
                "equation": " + ".join(r["equation"] for r in chosen),
                "member_reactions": [{"id": r["id"], "equation": r["equation"]} for r in chosen],
                "reactant_atom_types": _pad_atom_types(react_atoms, target_size),
                "product_atom_types": _pad_atom_types(prod_atoms, target_size),
                "reactant_adjacency": _pad_matrix(react_adj, target_size),
                "product_adjacency": _pad_matrix(prod_adj, target_size),
                "atom_count": len(react_atoms),
            }
        )

        for r in chosen:
            if r in unused:
                unused.remove(r)

    # Append the original grouped (shared-reactant) matrices as well.
    if include_grouped:
        entries.extend(build_grouped_training_set(target_size=target_size))
    return entries


def build_training_samples(
    n: int = 100,
    target_size: int = PAD_SIZE,
    seed: int = 0,
    exclude_ids: Optional[Set[int]] = None,
) -> List[Dict]:
    """Build training samples using the same packing strategy as validation."""
    return build_validation_set(n=n, target_size=target_size, seed=seed, include_grouped=False, exclude_ids=exclude_ids)


def save_validation_json(path: str, n: int = 10, target_size: int = PAD_SIZE, seed: int = 0) -> None:
    graphs = build_validation_set(n=n, target_size=target_size, seed=seed)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(graphs, fp, indent=2)


def _format_matrix(matrix: List[List[int]]) -> str:
    return "\n".join(" ".join(str(x) for x in row) for row in matrix)


def save_all_as_txt(path: str, target_reaction_size: int = PAD_SIZE, target_group_size: int = PAD_SIZE) -> None:
    """Write all reactions and grouped padded reactions to a text file."""
    reactions = build_reaction_graphs(target_size=target_reaction_size)
    groups = build_grouped_training_set(target_size=target_group_size)
    validations = build_validation_set(n=10, target_size=target_group_size)
    lines: List[str] = []

    lines.append(f"=== Individual reactions (padded to {target_reaction_size} atoms) ===")
    for entry in reactions:
        lines.append(f"\nID {entry['id']}: {entry['equation']}")
        lines.append("Reactant atom types: " + " ".join(entry["reactant_atom_types"]))
        lines.append("Reactant adjacency:")
        lines.append(_format_matrix(entry["reactant_adjacency"]))
        lines.append("Product atom types: " + " ".join(entry["product_atom_types"]))
        lines.append("Product adjacency:")
        lines.append(_format_matrix(entry["product_adjacency"]))

    lines.append(f"\n=== Grouped (shared-reactant) graphs padded to {target_group_size} atoms ===")
    for g in groups:
        lines.append(
            f"\nGroup {g['group_id']} signature {g['reactant_signature']}, "
            f"group_copies={g['group_copies']}, atoms={g['group_atom_count']}"
        )
        lines.append("Reactant atom types: " + " ".join(g["reactant_atom_types"]))
        lines.append("Reactant adjacency:")
        lines.append(_format_matrix(g["reactant_adjacency"]))
        lines.append("Combined product atom types: " + " ".join(g["product_atom_types"]))
        lines.append("Combined product adjacency:")
        lines.append(_format_matrix(g["product_adjacency"]))
        lines.append("Member reactions (order preserved in tiling):")
        for member in g["member_reactions"]:
            lines.append(f"  ID {member['id']}: {member['equation']}")

    lines.append(f"\n=== Validation samples (padded to {target_group_size} atoms) ===")
    for entry in validations:
        lines.append(f"\nEquation(s): {entry['equation']}")
        lines.append("Member reactions: " + ", ".join(f"{m['id']}" for m in entry["member_reactions"]))
        lines.append(f"Atom count (unpadded): {entry['atom_count']}")
        lines.append("Reactant atom types: " + " ".join(entry["reactant_atom_types"]))
        lines.append("Reactant adjacency:")
        lines.append(_format_matrix(entry["reactant_adjacency"]))
        lines.append("Product atom types: " + " ".join(entry["product_atom_types"]))
        lines.append("Product adjacency:")
        lines.append(_format_matrix(entry["product_adjacency"]))

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))


if __name__ == "__main__":
    graphs = build_reaction_graphs(target_size=PAD_SIZE)
    print(f"Padded adjacency size: {PAD_SIZE} atoms")
    print(f"Built {len(graphs)} reactions. Example entry (row-wise adjacency):")
    example = graphs[0]
    print(f"  ID {example['id']}: {example['equation']}")
    print("  Reactant adjacency:")
    for row in example["reactant_adjacency"]:
        print("   ", row)
    print("  Product adjacency:")
    for row in example["product_adjacency"]:
        print("   ", row)

    grouped = build_grouped_training_set()
    print("\nGrouped (shared-reactant) inputs padded to 12 atoms:")
    print(f"Built {len(grouped)} groups. Example group (row-wise adjacency):")
    g_ex = grouped[0]
    print(f"  Group {g_ex['group_id']}, reactant signature: {g_ex['reactant_signature']}")
    print("  Reactant adjacency:")
    for row in g_ex["reactant_adjacency"]:
        print("   ", row)
    print("  Combined product adjacency:")
    for row in g_ex["product_adjacency"]:
        print("   ", row)
    print("  Member reactions order:", [m["id"] for m in g_ex["member_reactions"]])

    out_txt = "reaction_matrices.txt"
    save_all_as_txt(out_txt)
    print(f"\nWrote all reactions and grouped padded matrices to {out_txt}")
