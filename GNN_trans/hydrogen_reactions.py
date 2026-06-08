from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SpeciesGraph:
    atom_types: List[str]
    adjacency: List[List[int]]


# Bond order: 0 = no bond, 1 = single, 2 = double, 3 = triple (unused here).
SPECIES: Dict[str, SpeciesGraph] = {
    "H": SpeciesGraph(atom_types=["H"], adjacency=[[0]]),
    "O": SpeciesGraph(atom_types=["O"], adjacency=[[0]]),
    "H2": SpeciesGraph(
        atom_types=["H", "H"],
        adjacency=[
            [0, 1],
            [1, 0],
        ],
    ),
    "O2": SpeciesGraph(
        atom_types=["O", "O"],
        adjacency=[
            [0, 2],
            [2, 0],
        ],
    ),
    "OH": SpeciesGraph(
        atom_types=["O", "H"],
        adjacency=[
            [0, 1],
            [1, 0],
        ],
    ),
    "HO2": SpeciesGraph(
        atom_types=["O", "O", "H"],
        adjacency=[
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
    ),
    "H2O": SpeciesGraph(
        atom_types=["O", "H", "H"],
        adjacency=[
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
    ),
    "H2O2": SpeciesGraph(
        atom_types=["O", "O", "H", "H"],
        adjacency=[
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
    ),
}


REACTIONS = [
    {"id": 1, "equation": "H + O2 <=> OH + O"},
    {"id": 2, "equation": "H2 + O <=> OH + H"},
    {"id": 3, "equation": "H2 + OH <=> H2O + H"},
    {"id": 4, "equation": "H2O + O <=> 2 OH"},
    {"id": 5, "equation": "2 H + M <=> H2 + M"},
    {"id": 6, "equation": "H + OH + M <=> H2O + M"},
    {"id": 7, "equation": "2 O + M <=> O2 + M"},
    {"id": 8, "equation": "H + O + M <=> OH + M"},
    {"id": 9, "equation": "O + OH + M <=> HO2 + M"},
    {"id": 10, "equation": "H + O2 (+ M) <=> HO2 (+ M)"},
    {"id": 11, "equation": "HO2 + H <=> 2 OH"},
    {"id": 12, "equation": "HO2 + H <=> H2 + O2"},
    {"id": 13, "equation": "HO2 + H <=> H2O + O"},
    {"id": 14, "equation": "HO2 + O <=> OH + O2"},
    {"id": 15, "equation": "HO2 + OH <=> H2O + O2"},
    {"id": 16, "equation": "2 OH (+ M) <=> H2O2 (+ M)"},
    {"id": 17, "equation": "2 HO2 <=> H2O2 + O2"},
    {"id": 18, "equation": "H2O2 + H <=> HO2 + H2"},
    {"id": 19, "equation": "H2O2 + H <=> H2O + OH"},
    {"id": 20, "equation": "H2O2 + OH <=> H2O + HO2"},
    {"id": 21, "equation": "H2O2 + O <=> HO2 + OH"},
]
