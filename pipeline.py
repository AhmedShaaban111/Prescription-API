"""
pipeline.py — Glue between prescription_agent.py and drug_pipeline.py
======================================================================

v4 — JSON output:
  - كل run بيحفظ ملف JSON كامل بجانب الـ terminal output
  - الـ JSON فيه: OCR data + matching results في object واحد
  - اسم الملف: <image_stem>_result.json  (نفس اسم الصورة)
  - ممكن تضيف --json-only لو مش عاوز الـ terminal table
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from prescription_agent import analyze_prescription
from drug_pipeline import build_index, match_many, print_results


def clean_drug_name(name: str) -> str:
    """
    Strip dose/strength info that was written after the drug name.

    Examples:
      "Sulrenta 25T"     → "Sulrenta"
      "Esomepraz 1×40"   → "Esomepraz"
      "Ciprafen (500)"   → "Ciprafen"
      "Gliben5"          → "Gliben"   (no space before digit)
      "Gaviscon"         → "Gaviscon" (unchanged)
    """
    name = re.sub(r'\s*[\(\[].*',              '', name)
    name = re.sub(r'\s+[\d×xX].*',             '', name)
    name = re.sub(r'(?<=[a-zA-Z])[\d×xX].*',   '', name)
    name = re.sub(r'\s+\d+\s*(?:mg|ml|mcg|iu|g)\b.*', '', name,
                  flags=re.IGNORECASE)
    return name.strip()


def build_meta(medicines: list[dict], cleaned_names: list[str]) -> dict[str, dict]:
    meta = {}
    for med, clean in zip(medicines, cleaned_names):
        meta[clean] = {
            "dose":      med.get("dose"),
            "frequency": med.get("frequency"),
            "duration":  med.get("duration"),
            "route":     med.get("route"),
            "notes":     med.get("notes"),
        }
    return meta


def build_json_output(
    image_path:   str,
    prescription: dict,
    cleaned_names: list[str],
    medicines:    list[dict],
    meta:         dict,
    results:      dict,
) -> dict:
    """
    Build the final JSON object combining OCR + matching results.

    Structure:
    {
      "generated_at": "ISO timestamp",
      "image":        "filename",
      "ocr": {
        "doctor":     ...,
        "patient":    ...,
        "date":       ...,
        "confidence": ...,
        "raw_text":   ...
      },
      "medicines": [
        {
          "ocr_name":        "as written on prescription",
          "cleaned_name":    "after clean_drug_name()",
          "dose":            ...,
          "frequency":       ...,
          "duration":        ...,
          "route":           ...,
          "notes":           ...,
          "read_confidence": "high|medium|low",
          "corrected_from":  "original name if local dict was applied",
          "matching": {
            "input_type":  "exact|fuzzy|misspelling|...",
            "active":      "INN/generic name",
            "category":    "rx|otc|supplement|other",
            "confidence":  "EXACT|HIGH|MEDIUM|LOW|NOT_FOUND",
            "note":        "human-readable resolution note",
            "matches":     ["DB product 1", "DB product 2", ...]
          }
        }
      ],
      "summary": {
        "total":     8,
        "found":     6,
        "not_found": 2
      }
    }
    """
    med_lookup = {m.get("name", "").strip(): m for m in medicines}

    medicines_json = []
    for clean in cleaned_names:
        # Find the original medicine dict (match by cleaned name back to raw)
        raw_med = next(
            (m for m in medicines
             if clean_drug_name(m.get("name", "")) == clean),
            {}
        )
        match_result = results.get(clean, {})

        entry = {
            "ocr_name":        raw_med.get("corrected_from") or raw_med.get("name", clean),
            "cleaned_name":    clean,
            "dose":            raw_med.get("dose"),
            "frequency":       raw_med.get("frequency"),
            "duration":        raw_med.get("duration"),
            "route":           raw_med.get("route"),
            "notes":           raw_med.get("notes"),
            "read_confidence": raw_med.get("read_confidence", "high"),
            "corrected_from":  raw_med.get("corrected_from"),   # None if no correction
            "matching": {
                "input_type": match_result.get("input_type"),
                "active":     match_result.get("active"),
                "category":   match_result.get("category"),
                "confidence": match_result.get("confidence"),
                "note":       match_result.get("note"),
                "matches":    match_result.get("matches", []),
            },
        }
        medicines_json.append(entry)

    found    = sum(1 for v in results.values() if v.get("confidence") != "NOT_FOUND")
    notfound = len(results) - found

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image":        Path(image_path).name,
        "ocr": {
            "doctor":     prescription.get("doctor"),
            "patient":    prescription.get("patient"),
            "date":       prescription.get("date"),
            "confidence": prescription.get("confidence"),
            "raw_text":   prescription.get("raw_text", ""),
        },
        "medicines": medicines_json,
        "summary": {
            "total":     len(results),
            "found":     found,
            "not_found": notfound,
        },
    }


def run(image_path: str, xlsx_path: str, k: int, json_only: bool = False):
    print(f"\nAnalyzing prescription image...")

    # ── Step 1: OCR ───────────────────────────────────────────────────────
    try:
        prescription = analyze_prescription(image_path)
    except Exception as e:
        print(f"[!] Failed to analyze image: {e}")
        sys.exit(1)

    # ── Print prescription header ─────────────────────────────────────────
    print()
    if prescription.get("doctor"):  print(f"  Doctor  : {prescription['doctor']}")
    if prescription.get("patient"): print(f"  Patient : {prescription['patient']}")
    if prescription.get("date"):    print(f"  Date    : {prescription['date']}")
    conf_map = {"high": "✅ Clear", "medium": "⚠️  Moderate", "low": "❌ Unclear"}
    print(f"  OCR     : {conf_map.get(prescription.get('confidence', ''), '')}")

    medicines = prescription.get("medicines", [])
    if not medicines:
        print("No medicines found in image.")
        return

    # ── Step 2: Clean names ───────────────────────────────────────────────
    raw_names     = [m["name"] for m in medicines if m.get("name")]
    cleaned_names = [clean_drug_name(n) for n in raw_names]

    print(f"\nExtracted medicines ({len(raw_names)}):")
    _rc_icon = {"high": "✅", "medium": "⚠️ ", "low": "❌"}
    for raw, clean, med in zip(raw_names, cleaned_names, medicines):
        rc   = (med.get("read_confidence") or "high").lower()
        icon = _rc_icon.get(rc, "")
        diff = f"  → '{clean}'" if raw.upper() != clean.upper() else ""
        print(f"  {icon} {raw}{diff}")

    # ── Step 3: Build / load DB index ─────────────────────────────────────
    build_index(xlsx_path, force=False)

    # ── Step 4: Build prescription metadata (dose/route/freq) ────────────
    meta = build_meta(medicines, cleaned_names)

    # ── Step 5: Match ─────────────────────────────────────────────────────
    results = match_many(cleaned_names, k=k, meta=meta)

    # ── Step 6: Terminal table (unless --json-only) ───────────────────────
    if not json_only:
        print_results(results, prescription_meta=meta)

    # ── Step 7: Save JSON ─────────────────────────────────────────────────
    output = build_json_output(
        image_path, prescription, cleaned_names, medicines, meta, results
    )
    json_path = Path(image_path).stem + "_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON saved → {json_path}\n")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match drugs from a handwritten prescription against a pharmacy database."
    )
    parser.add_argument("image",             help="Prescription image path")
    parser.add_argument("--xlsx", default="Book1.xlsx",
                        help="Excel database file (default: Book1.xlsx)")
    parser.add_argument("--k",    type=int, default=5,
                        help="Max matches per drug (default: 5)")
    parser.add_argument("--json-only", action="store_true",
                        help="Skip terminal table, only save JSON")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: '{args.image}' not found.")
        sys.exit(1)

    run(args.image, args.xlsx, args.k, json_only=args.json_only)