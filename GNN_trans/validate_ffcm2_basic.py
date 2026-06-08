"""Basic validation for FFCM2 graphs: atom counts, valence, and reaction conservation."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from FFCM2_reactions import SPECIES, REACTIONS


MAX_VALENCE = {
    "H": 1,
    "O": 2,
    "N": 3,
    "C": 4,
    "S": 2,
    "Cl": 1,
    "Br": 1,
    "F": 1,
    "I": 1,
    "He": 0,
    "Ne": 0,
    "Ar": 0,
}


def species_element_count(atom_types: List[str]) -> Counter:
    return Counter(atom_types)


def check_valence(atom_types: List[str], adjacency: List[List[int]]) -> List[Tuple[int, str, int, int]]:
    """Return list of (index, element, valence, max_valence) violations."""
    violations: List[Tuple[int, str, int, int]] = []
    n = len(atom_types)
    for i in range(n):
        elem = atom_types[i]
        max_v = MAX_VALENCE.get(elem)
        if max_v is None:
            continue
        valence = sum(int(b) for b in adjacency[i])
        if valence > max_v:
            violations.append((i, elem, valence, max_v))
    return violations


def _clean_term(term: str) -> str:
    t = term.strip()
    t = t.replace("(+ M)", "").replace("(+M)", "")
    t = t.replace("+ M", "").replace("+M", "")
    return " ".join(t.split())


def _parse_side(side: str) -> List[Tuple[str, int]]:
    parts = [p.strip() for p in side.split("+")]
    species: List[Tuple[str, int]] = []
    for part in parts:
        part = _clean_term(part)
        if not part or part == "M":
            continue
        tokens = part.split()
        if len(tokens) == 1:
            coeff = 1
            name = tokens[0]
        else:
            try:
                coeff = int(tokens[0])
                name = " ".join(tokens[1:])
            except ValueError:
                coeff = 1
                name = part
        name = name.strip()
        if name == "M":
            continue
        species.append((name, coeff))
    return species


def parse_equation(eqn: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if "<=>" in eqn:
        left, right = eqn.split("<=>", 1)
    elif "=>" in eqn:
        left, right = eqn.split("=>", 1)
    elif "=" in eqn:
        left, right = eqn.split("=", 1)
    else:
        raise ValueError(f"Unrecognized equation: {eqn}")
    return _parse_side(left), _parse_side(right)


def reaction_atom_balance(
    reactants: List[Tuple[str, int]],
    products: List[Tuple[str, int]],
) -> Tuple[Counter, Counter, List[str]]:
    missing: List[str] = []
    r_count: Counter = Counter()
    p_count: Counter = Counter()

    for name, coeff in reactants:
        graph = SPECIES.get(name)
        if graph is None:
            missing.append(name)
            continue
        for elem, count in species_element_count(graph.atom_types).items():
            r_count[elem] += count * coeff

    for name, coeff in products:
        graph = SPECIES.get(name)
        if graph is None:
            missing.append(name)
            continue
        for elem, count in species_element_count(graph.atom_types).items():
            p_count[elem] += count * coeff

    return r_count, p_count, missing


def validate_species() -> None:
    total = 0
    valence_violations: Dict[str, List[Tuple[int, str, int, int]]] = {}
    for name, graph in SPECIES.items():
        total += 1
        violations = check_valence(graph.atom_types, graph.adjacency)
        if violations:
            valence_violations[name] = violations

    print(f"Species checked: {total}")
    print(f"Species with valence violations: {len(valence_violations)}")
    for name, violations in list(valence_violations.items())[:20]:
        detail = ", ".join(
            f"{idx}:{elem} valence {val} > {max_v}" for idx, elem, val, max_v in violations
        )
        print(f"  {name}: {detail}")


def validate_reactions() -> None:
    imbalance = []
    missing_species = []

    for entry in REACTIONS:
        eqn = entry["equation"]
        reactants, products = parse_equation(eqn)
        r_count, p_count, missing = reaction_atom_balance(reactants, products)
        if missing:
            missing_species.append((entry["id"], eqn, missing))
            continue
        if r_count != p_count:
            imbalance.append((entry["id"], eqn, r_count, p_count))

    print(f"Reactions checked: {len(REACTIONS)}")
    print(f"Reactions with missing species: {len(missing_species)}")
    for rid, eqn, missing in missing_species[:20]:
        print(f"  {rid}: {eqn} missing={missing}")
    print(f"Reactions with atom imbalance: {len(imbalance)}")
    for rid, eqn, r_count, p_count in imbalance[:20]:
        print(f"  {rid}: {eqn}")
        print(f"    reactants: {dict(r_count)}")
        print(f"    products : {dict(p_count)}")


def main() -> None:
    print("== Species Valence Check ==")
    validate_species()
    print("\n== Reaction Atom Conservation ==")
    validate_reactions()


if __name__ == "__main__":
    main()
