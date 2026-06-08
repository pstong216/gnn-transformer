"""Generate a SMILES map for FFCM-2 species using PubChem InChI lookup."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib import error, request
from urllib.parse import quote

# Species list sourced from Stanford FFCM-2 table:
# https://web.stanford.edu/group/haiwanglab/FFCM2/docs/TrialModel/Species/
SPECIES_TABLE: List[Tuple[str, str, str]] = [
    ("H", "12385-13-6", "InChI=1S/H"),
    ("H2", "1333-74-0", "InChI=1S/H2/h1H"),
    ("O", "17778-80-2", "InChI=1S/O"),
    ("O2", "7782-44-7", "InChI=1S/O2/c1-2"),
    ("OH", "3352-57-6", "InChI=1S/HO/h1H"),
    ("H2O", "7732-18-5", "InChI=1S/H2O/h1H2"),
    ("HO2", "3170-83-0", "InChI=1S/HO2/c1-2/h1H"),
    ("H2O2", "7722-84-1", "InChI=1S/H2O2/c1-2/h1-2H"),
    ("HE", "7440-59-7", "InChI=1S/He"),
    ("AR", "7440-37-1", "InChI=1S/Ar"),
    ("N2", "7727-37-9", "InChI=1S/N2/c1-2"),
    ("C", "7440-44-0", "InChI=1S/C"),
    ("CH", "3315-37-5", "InChI=1S/CH/h1H"),
    ("CH2", "2465-56-7", "InChI=1S/CH2/h1H2"),
    ("CH2(S)", "2465-56-7", "SINGLET_InChI=1S/CH2/h1H2"),
    ("CH3", "2229-07-4", "InChI=1S/CH3/h1H3"),
    ("CH4", "74-82-8", "InChI=1S/CH4/h1H4"),
    ("CO", "630-08-0", "InChI=1S/CO/c1-2"),
    ("CO2", "124-38-9", "InChI=1S/CO2/c2-1-3"),
    ("HCO", "2597-44-6", "InChI=1S/CHO/c1-2/h1H"),
    ("CH2O", "50-00-0", "InChI=1S/CH2O/c1-2/h1H2"),
    ("CH2OH", "2597-43-5", "InChI=1S/CH3O/c1-2/h2H,1H2"),
    ("CH3O", "2143-68-2", "InChI=1S/CH3O/c1-2/h1H3"),
    ("CH3OH", "67-56-1", "InChI=1S/CH4O/c1-2/h2H,1H3"),
    ("C2H", "2122-48-7", "InChI=1S/C2H/c1-2/h1H"),
    ("C2H2", "74-86-2", "InChI=1S/C2H2/c1-2/h1-2H"),
    ("C2H3", "2669-89-8", "InChI=1S/C2H3/c1-2/h1H,2H2"),
    ("C2H4", "74-85-1", "InChI=1S/C2H4/c1-2/h1-2H2"),
    ("C2H5", "2025-56-1", "InChI=1S/C2H5/c1-2/h1H2,2H3"),
    ("C2H6", "74-84-0", "InChI=1S/C2H6/c1-2/h1-2H3"),
    ("HCCO", "51095-15-9", "InChI=1S/C2HO/c1-2-3/h1H"),
    ("CH2CO", "463-51-4", "InChI=1S/C2H2O/c1-2-3/h1H2"),
    ("CH2CHO", "6912-06-7", "InChI=1S/C2H3O/c1-2-3/h2H,1H2"),
    ("CH3CHO", "75-07-0", "InChI=1S/C2H4O/c1-2-3/h2H,1H3"),
    ("CH3CO", "3170-69-2", "InChI=1S/C2H3O/c1-2-3/h1H3"),
    ("H2CC", "2143-69-3", "InChI=1S/C2H2/c1-2/h1H2"),
    ("CH3O2", "2143-58-0", "InChI=1S/CH3O2/c1-3-2/h1H3"),
    ("CH3OOH", "3031-73-0", "InChI=1S/CH4O2/c1-3-2/h2H,1H3"),
    ("C2H5O2", "3170-61-4", "InChI=1S/C2H5O2/c1-2-4-3/h2H2,1H3"),
    ("C2H5OOH", "3031-74-1", "InChI=1S/C2H6O2/c1-2-4-3/h3H,2H2,1H3"),
    ("C2H2OH", "N/A", "InChI=1S/C2H3O/c1-2-3/h1-3H"),
    ("C2H3OH", "557-75-5", "InChI=1S/C2H4O/c1-2-3/h2-3H,1H2"),
    ("C2H4O", "75-07-0", "InChI=1S/C2H4O/c1-2-3/h2H,1H3"),
    ("C2H5OH", "64-17-5", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"),
    ("C2H4OH", "4422-54-2", "InChI=1S/C2H5O/c1-2-3/h3H,1-2H2"),
    ("CH3CHOH", "2348-46-1", "InChI=1S/C2H5O/c1-2-3/h2-3H,1H3"),
    ("C3H8", "74-98-6", "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3"),
    ("NC3H7", "2143-61-5", "InChI=1S/C3H7/c1-3-2/h1,3H2,2H3"),
    ("IC3H7", "2025-55-0", "InChI=1S/C3H7/c1-3-2/h3H,1-2H3"),
    ("C3H6", "115-07-1", "InChI=1S/C3H6/c1-3-2/h3H,1H2,2H3"),
    ("C3H5", "1981-80-2", "InChI=1S/C3H5/c1-3-2/h3H,1-2H2"),
    ("CH3CCH2", "N/A", "InChI=1S/C3H5/c1-3-2/h1H2,2H3"),
    ("AC3H4", "463-49-0", "InChI=1S/C3H4/c1-3-2/h1-2H2"),
    ("PC3H4", "74-99-7", "InChI=1S/C3H4/c1-3-2/h1H,2H3"),
    ("C3H3", "2932-78-7", "InChI=1S/C3H3/c1-3-2/h1H,2H2"),
    ("C2H5CHO", "123-38-6", "InChI=1S/C3H6O/c1-2-3-4/h3H,2H2,1H3"),
    ("CH3COCH3", "67-64-1", "InChI=1S/C3H6O/c1-3(2)4/h1-2H3"),
    ("CH3COCH2", "3122-07-4", "InChI=1S/C3H5O/c1-3(2)4/h1H2,2H3"),
    ("C2H3CHO", "107-02-8", "InChI=1S/C3H4O/c1-2-3-4/h2-3H,1H2"),
    ("C3H5OH", "107-18-6", "InChI=1S/C3H6O/c1-2-3-4/h2,4H,1,3H2"),
    ("NC3H7O2", "42953-38-8", "InChI=1S/C3H7O2/c1-2-3-5-4/h2-3H2,1H3"),
    ("NC3H7OOH", "6068-96-8", "InChI=1S/C3H8O2/c1-2-3-5-4/h4H,2-3H2,1H3"),
    ("IC3H7O2", "N/A", "InChI=1S/C3H7O2/c1-3(2)5-4/h3H,1-2H3"),
    ("IC3H7OOH", "3031-75-2", "InChI=1S/C3H8O2/c1-3(2)5-4/h3-4H,1-2H3"),
    ("C4H2", "460-12-8", "InChI=1S/C4H2/c1-3-4-2/h1-2H"),
    ("NC4H3", "N/A", "InChI=1S/C4H3/c1-3-4-2/h3H,1H2"),
    ("IC4H3", "22112-56-7", "InChI=1S/C4H3/c1-3-4-2/h1H,2H2"),
    ("C4H4", "689-97-4", "InChI=1S/C4H4/c1-3-4-2/h1,4H,2H2"),
    ("NC4H5", "N/A", "InChI=1S/C4H5/c1-3-4-2/h3H2,1H3"),
    ("IC4H5", "3315-42-2", "InChI=1S/C4H5/c1-3-4-2/h1,4H,2H3"),
    ("C4H5-2", "82252-88-8", "InChI=1S/C4H5/c1-3-4-2/h1H2,2H3"),
    ("C4H6", "106-99-0", "InChI=1S/C4H6/c1-3-4-2/h3-4H,1-2H2"),
    ("C4H612", "590-19-2", "InChI=1S/C4H6/c1-3-4-2/h4H,1H2,2H3"),
    ("C4H6-2", "503-17-3", "InChI=1S/C4H6/c1-3-4-2/h1-2H3"),
    ("C4H7", "65338-31-0", "InChI=1S/C4H7/c1-3-4-2/h3-4H,1H2,2H3"),
    ("IC4H7", "15157-95-6", "InChI=1S/C4H7/c1-4(2)3/h1-2H2,3H3"),
    ("IC4H7-1", "N/A", "InChI=1S/C4H7/c1-4(2)3/h1H,2-3H3"),
    ("C4H81", "106-98-9", "InChI=1S/C4H8/c1-3-4-2/h3H,1,4H2,2H3"),
    ("C4H82", "107-01-7", "InChI=1S/C4H8/c1-3-4-2/h3-4H,1-2H3"),
    ("IC4H8", "115-11-7", "InChI=1S/C4H8/c1-4(2)3/h1H2,2-3H3"),
    ("NC4H9", "2492-36-6", "InChI=1S/C4H9/c1-3-4-2/h1,3-4H2,2H3"),
    ("SC4H9", "2348-55-2", "InChI=1S/C4H9/c1-3-4-2/h3H,4H2,1-2H3"),
    ("IC4H9", "4630-45-9", "InChI=1S/C4H9/c1-4(2)3/h4H,1H2,2-3H3"),
    ("TC4H9", "1605-73-8", "InChI=1S/C4H9/c1-4(2)3/h1-3H3"),
    ("C4H10", "106-97-8", "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3"),
    ("IC4H10", "75-28-5", "InChI=1S/C4H10/c1-4(2)3/h4H,1-3H3"),
    ("H2C4O", "63766-91-6", "InChI=1S/C4H2O/c1-2-3-4-5/h1H2"),
    ("CH2CHCHCHO", "N/A", "InChI=1S/C4H5O/c1-2-3-4-5/h2-4H,1H2"),
    ("CH3CHCHCO", "N/A", "InChI=1S/C4H5O/c1-2-3-4-5/h2-3H,1H3"),
    ("CH3CHCHCHO", "4170-30-3", "InChI=1S/C4H6O/c1-2-3-4-5/h2-4H,1H3"),
    ("C3H7CHO", "123-72-8", "InChI=1S/C4H8O/c1-2-3-4-5/h4H,2-3H2,1H3"),
    ("IC3H7CHO", "78-84-2", "InChI=1S/C4H8O/c1-4(2)3-5/h3-4H,1-2H3"),
    ("C2H5COCH3", "78-93-3", "InChI=1S/C4H8O/c1-3-4(2)5/h3H2,1-2H3"),
    ("C2H3COCH3", "78-94-4", "InChI=1S/C4H6O/c1-3-4(2)5/h3H,1H2,2H3"),
    ("OH*", "3352-57-6", "EXCITED_InChI=1S/HO/h1H"),
    ("CH*", "3315-37-5", "EXCITED_InChI=1S/CH/h1H"),
]

SMILES_MAP_PATH = Path(__file__).with_name("FFCM2_smiles_map.json")

PUBCHEM_CAS_URL_TEMPLATE = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RN/{cas}/"
    "property/{properties}/JSON"
)
PUBCHEM_INCHI_URL_TEMPLATE = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/"
    "property/{properties}/JSON?inchi={inchi}"
)

PROPERTY_PRIORITY = (
    "CanonicalSMILES",
    "IsomericSMILES",
    "SMILES",
    "ConnectivitySMILES",
)

PROPERTY_QUERY = ",".join(dict.fromkeys((*PROPERTY_PRIORITY, "InChI")))


def _normalize_inchi(value: str) -> str | None:
    if not value:
        return None
    normalized = value.strip()
    if not normalized or normalized.upper() == "N/A":
        return None
    marker = normalized.find("InChI=")
    if marker >= 0:
        normalized = normalized[marker:]
    return normalized if normalized.startswith("InChI=") else None


def _normalize_species_rows(
    rows: Iterable[Tuple[str, str, str]]
) -> List[Tuple[str, str | None, str | None]]:
    normalized: List[Tuple[str, str | None, str | None]] = []
    for abbr, cas, inchi in rows:
        cas_norm = cas.strip() if cas and cas.upper() != "N/A" else None
        inchi_norm = _normalize_inchi(inchi)
        normalized.append((abbr, cas_norm, inchi_norm))
    return normalized


SPECIES_INCHI: List[Tuple[str, str | None, str | None]] = _normalize_species_rows(SPECIES_TABLE)


def _pubchem_inchi_url(inchi: str) -> str:
    return PUBCHEM_INCHI_URL_TEMPLATE.format(
        inchi=quote(inchi, safe=""),
        properties=PROPERTY_QUERY,
    )


def _pubchem_cas_url(cas: str) -> str:
    return PUBCHEM_CAS_URL_TEMPLATE.format(
        cas=quote(cas, safe=""),
        properties=PROPERTY_QUERY,
    )


def _request_pubchem_json(
    url: str,
    *,
    context: str,
    timeout: int = 10,
    max_retries: int = 3,
    backoff: float = 0.5,
) -> Dict[str, Any] | None:
    attempt = 0

    while True:
        try:
            with request.urlopen(url, timeout=timeout) as response:
                payload = response.read()
            break
        except error.HTTPError as exc:
            if exc.code == 404:
                return None
            attempt += 1
            if attempt >= max_retries:
                raise RuntimeError(f"PubChem request failed for {context}: {exc}") from exc
            time.sleep(backoff * (2 ** (attempt - 1)))
        except error.URLError as exc:
            attempt += 1
            if attempt >= max_retries:
                raise RuntimeError(f"PubChem request failed for {context}: {exc}") from exc
            time.sleep(backoff * (2 ** (attempt - 1)))

    try:
        data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"PubChem response decode error for {context}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected PubChem payload for {context}")

    if "Fault" in data:
        fault = data["Fault"]
        message = fault.get("Message") if isinstance(fault, dict) else str(fault)
        raise RuntimeError(f"PubChem fault for {context}: {message}")
    if "Waiting" in data:
        raise RuntimeError(
            f"PubChem deferred response for {context}; asynchronous handling not implemented"
        )

    return data


def _extract_smiles(
    properties: Iterable[Dict[str, Any]],
    *,
    expected_inchi: str | None = None,
) -> tuple[str, str | None] | None:
    fallback: tuple[str, str | None] | None = None
    found_exact = False

    for entry in properties:
        if not isinstance(entry, dict):
            continue

        entry_inchi = entry.get("InChI") or entry.get("inchi")
        entry_inchi = str(entry_inchi).strip() if entry_inchi else None

        smiles_value = None
        for key in PROPERTY_PRIORITY:
            value = entry.get(key) or entry.get(key.lower())
            if value:
                smiles_value = str(value).strip()
                break

        if not smiles_value:
            continue

        record = (smiles_value, entry_inchi)

        if expected_inchi and entry_inchi:
            if entry_inchi == expected_inchi:
                found_exact = True
                return record

        if fallback is None:
            fallback = record

    if expected_inchi:
        return None if not found_exact else fallback
    return fallback


def fetch_smiles_for_inchi(
    inchi: str,
    *,
    expected_inchi: str | None = None,
    timeout: int = 10,
    max_retries: int = 3,
    backoff: float = 0.5,
) -> str | None:
    url = _pubchem_inchi_url(inchi)
    data = _request_pubchem_json(
        url,
        context=f"InChI {inchi}",
        timeout=timeout,
        max_retries=max_retries,
        backoff=backoff,
    )
    if data is None:
        return None

    properties = data.get("PropertyTable", {}).get("Properties", [])
    result = _extract_smiles(properties, expected_inchi=expected_inchi or inchi)
    return result[0] if result else None


def fetch_smiles_for_cas(
    cas: str,
    *,
    expected_inchi: str | None = None,
    timeout: int = 10,
    max_retries: int = 3,
    backoff: float = 0.5,
) -> str | None:
    url = _pubchem_cas_url(cas)
    data = _request_pubchem_json(
        url,
        context=f"CAS {cas}",
        timeout=timeout,
        max_retries=max_retries,
        backoff=backoff,
    )
    if data is None:
        return None

    properties = data.get("PropertyTable", {}).get("Properties", [])
    result = _extract_smiles(properties, expected_inchi=expected_inchi)
    return result[0] if result else None


def load_smiles_map(smiles_path: Path = SMILES_MAP_PATH) -> Dict[str, str]:
    if not smiles_path.exists():
        return {}
    with smiles_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str)}


def save_smiles_map(smiles_map: Dict[str, str], smiles_path: Path = SMILES_MAP_PATH) -> None:
    with smiles_path.open("w", encoding="utf-8") as fh:
        json.dump(smiles_map, fh, indent=2, sort_keys=True)


def build_species_smiles_map(
    species_rows: Iterable[Tuple[str, str | None, str | None]] = SPECIES_INCHI,
    *,
    smiles_path: Path = SMILES_MAP_PATH,
    request_delay: float = 0.2,
    verbose: bool = True,
) -> Dict[str, str]:
    """Build a full SMILES map, requiring both CAS and InChI to match."""
    smiles_map = load_smiles_map(smiles_path)
    updated: Dict[str, str] = {}

    for abbreviation, cas, inchi in species_rows:
        existing = smiles_map.get(abbreviation)
        if existing and existing != "MISSING":
            updated[abbreviation] = existing
            if verbose:
                print(f"[{abbreviation}] Loaded from SMILES map")
            continue

        if verbose:
            print(f"[{abbreviation}] Fetching from PubChem (CAS + InChI)...")

        if not cas or not inchi:
            if verbose:
                print(f"[{abbreviation}] Missing CAS or InChI; marking as MISSING")
            updated[abbreviation] = "MISSING"
            continue

        inchi_smiles = None
        cas_smiles = None

        try:
            inchi_smiles = fetch_smiles_for_inchi(inchi, expected_inchi=inchi)
        except RuntimeError as exc:
            if verbose:
                print(f"[{abbreviation}] InChI lookup failed: {exc}")
        finally:
            if request_delay:
                time.sleep(request_delay)

        try:
            cas_smiles = fetch_smiles_for_cas(cas, expected_inchi=inchi)
        except RuntimeError as exc:
            if verbose:
                print(f"[{abbreviation}] CAS lookup failed: {exc}")
        finally:
            if request_delay:
                time.sleep(request_delay)

        if not inchi_smiles or not cas_smiles:
            if verbose:
                print(f"[{abbreviation}] Lookup incomplete; marking as MISSING")
            updated[abbreviation] = "MISSING"
            continue

        if inchi_smiles != cas_smiles:
            if verbose:
                print(
                    f"[{abbreviation}] CAS/InChI mismatch: {inchi_smiles} vs {cas_smiles}"
                )
            updated[abbreviation] = "MISMATCH"
            continue

        updated[abbreviation] = inchi_smiles
        if verbose:
            print(f"[{abbreviation}] Resolved via CAS+InChI: {inchi_smiles}")

    save_smiles_map(updated, smiles_path)
    return updated


if __name__ == "__main__":
    smiles_map = build_species_smiles_map(verbose=True)
    print(f"\nSaved SMILES map to {SMILES_MAP_PATH.name} ({len(smiles_map)} entries)")
