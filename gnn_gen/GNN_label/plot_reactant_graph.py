from __future__ import annotations

import os
import torch

from hydrogen_adjacency import REACTIONS, combine_species, parse_equation, reorder_to_target
from plotting import plot_graph


# Mini script settings
REACTION_ID = 15
REACTION_REPEAT = 1
OUT_DIR = "reaction_dataset_prediction_rate1"


def _reaction_by_id(reaction_id: int) -> dict:
    for rxn in REACTIONS:
        if int(rxn.get("id", -1)) == int(reaction_id):
            return rxn
    valid = [int(r["id"]) for r in REACTIONS]
    raise ValueError(f"Unknown reaction id {reaction_id}. Valid ids: {valid}")


def main() -> None:
    rxn = _reaction_by_id(REACTION_ID)
    equation = str(rxn["equation"])
    reactants, products = parse_equation(equation)
    reactants_use = reactants * max(1, int(REACTION_REPEAT))
    products_use = products * max(1, int(REACTION_REPEAT))

    react_adj, react_atom_types = combine_species(reactants_use)
    prod_adj, prod_atom_types = combine_species(products_use)
    prod_adj_aligned = reorder_to_target(
        target_atoms=react_atom_types,
        source_atoms=prod_atom_types,
        source_adj=prod_adj,
        target_adj=react_adj,
    )

    react_adj_t = torch.tensor(react_adj, dtype=torch.float32)
    prod_adj_t = torch.tensor(prod_adj_aligned, dtype=torch.float32)

    os.makedirs(OUT_DIR, exist_ok=True)
    react_out = os.path.join(OUT_DIR, f"demo_reactant_reaction_{REACTION_ID}.png")
    prod_out = os.path.join(OUT_DIR, f"demo_product_reaction_{REACTION_ID}.png")

    title = f"Reactant Graph | Reaction {REACTION_ID} | x{max(1, int(REACTION_REPEAT))}"
    plot_graph(react_adj_t, react_atom_types, react_out, title)
    plot_graph(
        prod_adj_t,
        react_atom_types,
        prod_out,
        f"Product Graph (Aligned) | Reaction {REACTION_ID} | x{max(1, int(REACTION_REPEAT))}",
    )
    print(f"Saved reactant graph: {react_out}")
    print(f"Saved product graph:  {prod_out}")
    print(f"Equation: {equation}")


if __name__ == "__main__":
    main()
