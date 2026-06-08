"""Generate a Python module of species graphs from a SMILES map."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to parse FFCM2_model.yaml.txt") from exc


SMILES_MAP_PATH = Path(__file__).with_name("FFCM2_smiles_map.json")
FFCM2_YAML = Path(__file__).with_name("FFCM2_model.yaml.txt")
OUTPUT_PY = Path(__file__).with_name("FFCM2_reactions.py")


@dataclass(frozen=True)
class SpeciesGraph:
    atom_types: List[str]
    adjacency: List[List[int]]


def _load_smiles_map(path: Path = SMILES_MAP_PATH) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"SMILES map not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("SMILES map JSON is not a dict")
    return {k: v for k, v in data.items() if isinstance(v, str)}


def _smiles_to_graph(smiles: str, *, include_hs: bool = True) -> SpeciesGraph:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required to convert SMILES to graphs.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"RDKit failed to parse SMILES: {smiles}")
    if include_hs:
        mol = Chem.AddHs(mol)
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    n = len(atom_types)
    adj = [[0 for _ in range(n)] for _ in range(n)]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        if order in (1.5, 2.5):
            order = int(order - 0.5)
        else:
            order = int(round(order))
        adj[i][j] = order
        adj[j][i] = order
    return SpeciesGraph(atom_types=atom_types, adjacency=adj)


def _load_reactions(yaml_path: Path = FFCM2_YAML) -> List[Dict[str, str]]:
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    reactions = []
    for idx, entry in enumerate(data.get("reactions", []), start=1):
        eqn = entry.get("equation")
        if eqn:
            reactions.append({"id": idx, "equation": eqn})
    return reactions


def _write_python_module(
    species: Dict[str, SpeciesGraph],
    reactions: List[Dict[str, str]],
    output_path: Path = OUTPUT_PY,
) -> None:
    lines: List[str] = []
    lines.append("from dataclasses import dataclass\n")
    lines.append("from typing import Dict, List\n\n\n")
    lines.append("@dataclass(frozen=True)\n")
    lines.append("class SpeciesGraph:\n")
    lines.append("    atom_types: List[str]\n")
    lines.append("    adjacency: List[List[int]]\n\n\n")
    lines.append("SPECIES: Dict[str, SpeciesGraph] = {\n")
    for name, graph in species.items():
        lines.append(f"    {name!r}: SpeciesGraph(\n")
        lines.append(f"        atom_types={graph.atom_types!r},\n")
        lines.append("        adjacency=[\n")
        for row in graph.adjacency:
            lines.append(f"            {row},\n")
        lines.append("        ],\n")
        lines.append("    ),\n")
    lines.append("}\n\n\n")
    lines.append("REACTIONS = [\n")
    for entry in reactions:
        lines.append(f"    {{'id': {entry['id']}, 'equation': {entry['equation']!r}}},\n")
    lines.append("]\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def build_graphs() -> None:
    smiles_map = _load_smiles_map()
    species: Dict[str, SpeciesGraph] = {}
    missing: List[str] = []

    for name, smiles in smiles_map.items():
        if smiles in {"MISSING", "MISMATCH", ""}:
            print(f"[{name}] skipped (SMILES={smiles})")
            missing.append(name)
            continue
        try:
            species[name] = _smiles_to_graph(smiles, include_hs=True)
            print(f"[{name}] graph built")
        except Exception as exc:
            print(f"[{name}] graph build failed: {exc}")
            missing.append(name)

    reactions = _load_reactions()
    _write_python_module(species, reactions)
    print(f"Saved {len(species)} species to {OUTPUT_PY.name}")
    if missing:
        print(f"Missing graphs for {len(missing)} species: {', '.join(missing[:20])}")


if __name__ == "__main__":
    build_graphs()
