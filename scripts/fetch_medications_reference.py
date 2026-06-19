"""Build alethia/data/medications_reference.csv from the NLM RxNorm / RxClass APIs."""

# Source: U.S. National Library of Medicine RxNav web services (public, no API key).
#   RxNorm name -> RxCUI:        https://rxnav.nlm.nih.gov/REST/rxcui.json
#   ATC classes for an RxCUI:    https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json
# For each generic we resolve its RxCUI, fetch its ATC (WHO Anatomical Therapeutic Chemical)
# codes, and assign the level-1 anatomical class the codes most often fall under, so names
# and classes come from authoritative NLM data. Edit SEED_DRUGS to change coverage.
#
# Usage: python scripts/fetch_medications_reference.py [--out PATH]
# RxNorm/RxClass are in the U.S. public domain: https://www.nlm.nih.gov/research/umls/rxnorm/

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter

ATC_LEVEL1 = {
    "A": "Alimentary tract and metabolism",
    "B": "Blood and blood forming organs",
    "C": "Cardiovascular system",
    "D": "Dermatologicals",
    "G": "Genito-urinary system and sex hormones",
    "H": "Systemic hormonal preparations",
    "J": "Antiinfectives for systemic use",
    "L": "Antineoplastic and immunomodulating agents",
    "M": "Musculo-skeletal system",
    "N": "Nervous system",
    "P": "Antiparasitic products",
    "R": "Respiratory system",
    "S": "Sensory organs",
    "V": "Various",
}

# Widely used generics across therapeutic areas. Coverage is what the demo needs; the
# therapeutic class for each is fetched from ATC, not hard-coded here.
SEED_DRUGS = [
    "Paracetamol",
    "Ibuprofen",
    "Aspirin",
    "Diclofenac",
    "Naproxen",
    "Tramadol",
    "Amoxicillin",
    "Azithromycin",
    "Ciprofloxacin",
    "Doxycycline",
    "Metronidazole",
    "Ceftriaxone",
    "Cephalexin",
    "Clindamycin",
    "Trimethoprim",
    "Metformin",
    "Glibenclamide",
    "Gliclazide",
    "Insulin",
    "Sitagliptin",
    "Atorvastatin",
    "Simvastatin",
    "Rosuvastatin",
    "Amlodipine",
    "Losartan",
    "Telmisartan",
    "Enalapril",
    "Ramipril",
    "Bisoprolol",
    "Atenolol",
    "Hydrochlorothiazide",
    "Furosemide",
    "Spironolactone",
    "Warfarin",
    "Clopidogrel",
    "Heparin",
    "Rivaroxaban",
    "Omeprazole",
    "Pantoprazole",
    "Esomeprazole",
    "Ranitidine",
    "Ondansetron",
    "Salbutamol",
    "Montelukast",
    "Budesonide",
    "Fluticasone",
    "Theophylline",
    "Cetirizine",
    "Loratadine",
    "Levothyroxine",
    "Prednisolone",
    "Dexamethasone",
    "Hydrocortisone",
    "Sertraline",
    "Fluoxetine",
    "Amitriptyline",
    "Diazepam",
    "Lorazepam",
    "Levetiracetam",
    "Gabapentin",
    "Morphine",
    "Metoclopramide",
]

NAME_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name={name}&search=2"
ATC_URL = (
    "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"
    "?rxcui={rxcui}&relaSource=ATC"
)


def _get(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.load(resp)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)


def rxcui_for(name):
    data = _get(NAME_URL.format(name=urllib.parse.quote(name))) or {}
    ids = data.get("idGroup", {}).get("rxnormId") or []
    return ids[0] if ids else None


def atc_level1_for(rxcui):
    data = _get(ATC_URL.format(rxcui=rxcui)) or {}
    items = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
    letters = Counter()
    for it in items:
        mc = it.get("rxclassMinConceptItem", {})
        if mc.get("classType") == "ATC1-4":
            code = mc.get("classId", "")
            if code and code[0] in ATC_LEVEL1:
                letters[code[0]] += 1
    if not letters:
        return None
    return ATC_LEVEL1[letters.most_common(1)[0][0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="alethia/data/medications_reference.csv")
    args = ap.parse_args()

    rows = []
    for name in SEED_DRUGS:
        cui = rxcui_for(name)
        if not cui:
            print(f"  skip (no RxCUI): {name}", file=sys.stderr)
            continue
        cls = atc_level1_for(cui)
        if not cls:
            print(f"  skip (no ATC class): {name}", file=sys.stderr)
            continue
        rows.append((name, cls))
        print(f"{name}: {cls}", file=sys.stderr)
        time.sleep(0.2)

    rows.sort()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generic_name", "therapeutic_class"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} drugs to {args.out}")


if __name__ == "__main__":
    main()
