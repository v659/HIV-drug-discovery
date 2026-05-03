"""Fetch canonical SMILES from PubChem for FDA-approved HIV drugs by class.

Writes one file per mechanism class to src/drug_classes/{class}.txt
(one canonical SMILES per line). These become the reference sets for
the class-aware Tanimoto stacker features.

Drug names are sourced from FDA-approved single-agent HIV antiretrovirals
across 5 mechanism classes:
  - NRTI:  nucleoside/nucleotide RT inhibitors
  - NNRTI: non-nucleoside RT inhibitors
  - PI:    protease inhibitors
  - INSTI: integrase strand-transfer inhibitors
  - Entry: CCR5 antagonists, attachment inhibitors, capsid inhibitors

Excluded: peptide drugs (enfuvirtide, T-20) and antibody drugs (ibalizumab)
because their representations don't translate to small-molecule SMILES.

PubChem REST API has no auth requirement; rate-limited at 5 req/sec so we
sleep 0.3s between calls.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent / "drug_classes"

DRUGS_BY_CLASS = {
    "nrti": [
        "Zidovudine", "Lamivudine", "Stavudine", "Emtricitabine",
        "Didanosine", "Abacavir", "Tenofovir", "Zalcitabine",
    ],
    "nnrti": [
        "Nevirapine", "Efavirenz", "Etravirine", "Rilpivirine",
        "Doravirine", "Delavirdine",
    ],
    "pi": [
        "Saquinavir", "Ritonavir", "Indinavir", "Nelfinavir", "Amprenavir",
        "Lopinavir", "Atazanavir", "Fosamprenavir", "Tipranavir", "Darunavir",
    ],
    "insti": [
        "Raltegravir", "Elvitegravir", "Dolutegravir", "Bictegravir",
        "Cabotegravir",
    ],
    "entry": [
        "Maraviroc", "Fostemsavir", "Lenacapavir",
    ],
}

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/"
    "property/SMILES/JSON"
)


def fetch_smiles(name):
    url = PUBCHEM_URL.format(name=urllib.parse.quote(name))
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
            props = data["PropertyTable"]["Properties"][0]
            # PubChem renamed fields in 2024+: prefer SMILES (isomeric) over
            # ConnectivitySMILES (no stereo). CanonicalSMILES no longer exists.
            return props.get("SMILES") or props.get("ConnectivitySMILES")
    except Exception as e:
        print(f"  ! {name}: {type(e).__name__}: {e}")
        return None


def main():
    OUT_DIR.mkdir(exist_ok=True)
    summary = []
    all_records = []  # for lookup module: name + class + SMILES
    for cls, names in DRUGS_BY_CLASS.items():
        print(f"\nFetching {cls.upper()} ({len(names)} drugs)...")
        smiles_list = []
        for name in names:
            smi = fetch_smiles(name)
            if smi:
                smiles_list.append(smi)
                all_records.append({"name": name, "class": cls, "smiles": smi})
                print(f"  {name:20s} -> {smi}")
            time.sleep(0.3)
        path = OUT_DIR / f"{cls}.txt"
        path.write_text("\n".join(smiles_list) + "\n")
        summary.append((cls, len(smiles_list), len(names), path))

    # Also write a single JSON with name+class+SMILES for the lookup module.
    json_path = OUT_DIR / "all_drugs.json"
    json_path.write_text(json.dumps(all_records, indent=2))

    print("\n=== Summary ===")
    for cls, got, asked, path in summary:
        print(f"  {cls:6s}: {got}/{asked} drugs -> {path}")
    print(f"  Combined records: {len(all_records)} -> {json_path}")


if __name__ == "__main__":
    main()
