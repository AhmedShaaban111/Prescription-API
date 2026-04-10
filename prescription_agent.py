"""
prescription_agent.py — Handwritten Prescription OCR via Gemini
================================================================

v6 — Single strong pass + smart deduplication:

  الـ dual-run consensus كان بيضيّع أدوية لأنه كان بياخد
  الـ intersection مش الـ union.

  الحل الأبسط والأموثق:
    - Pass واحد بـ prompt قوي جداً يقرأ كل الأدوية
    - Deduplication بعده عشان يشيل التكرار
    - لو الـ OCR confidence منخفض → نشغّل pass تاني
      بس نحتفظ بالـ UNION مش الـ intersection
"""

import base64
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from rapidfuzz import fuzz
from openai import OpenAI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL          = os.getenv("MODEL", "gemini-2.0-flash")

if not GEMINI_API_KEY:
    raise ValueError("No GEMINI_API_KEY found in .env file!")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# ══════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an OCR specialist for handwritten medical prescriptions.

YOUR MOST IMPORTANT RULE: Extract EVERY SINGLE medicine written on the prescription.
Do NOT stop after 2-3 medicines. Read the ENTIRE prescription from top to bottom.
Count R/ or Rx markers — each one is a new medicine entry.

Drug name rules:
- Copy EXACTLY as written, letter by letter
- Do NOT replace with generic/brand equivalents
  ("Ciprafen" → "Ciprafen", NOT "Ciprofloxacin")
  ("Sulrenta" → "Sulrenta", NOT "Vortioxetine")
- Include ALL medicines even if handwriting is unclear

Dose rules:
- Number + unit only, strip T/Tab/Cap suffixes
  "25T" → "25mg",  "500mg" → "500mg",  "1×40" → "40mg"
- No unit written → assume "mg"

Frequency rules:
- Read ONLY what is physically written — do NOT assume from drug class
- Unclear → null
- bd/bid/مرتين/×2 = twice daily
- tds/tid/ثلاث مرات/×3 = three times daily
- od/مره/once = once daily
- qid/×4 = four times daily
- قبل الاكل = before meals    بعد الاكل = after meals
- ص م م = morning noon evening = three times daily

CRITICAL MEDICAL SAFETY RULES:
- Use the doctor's specialty or other drugs on the prescription as CONTEXT.
  e.g. if most drugs are GI drugs, prioritize GI drug interpretations.
- NEVER suggest a psychiatric/neurological drug (like Brintellix/vortioxetine)
  for a prescription that is clearly gastroenterological.
- If a word could be a misspelling of a COMMON drug, prefer the common drug
  over an obscure or unrelated one.
- If the handwriting is too ambiguous and no common drug fits → return null
  for the name rather than hallucinating a brand name.
- Prefer well-known trade names over generic INNs when the prescription clearly
  uses brand names.

Per-medicine read_confidence:
- "high"   → every letter clearly visible
- "medium" → mostly clear, minor uncertainty in 1-2 chars
- "low"    → unclear handwriting, best-guess attempt

Return ONLY valid JSON — make sure medicines array is COMPLETE:
{
  "doctor":   "name or null",
  "patient":  "name or null",
  "date":     "date or null",
  "medicines": [
    {
      "name":            "EXACT name as written",
      "dose":            "number+unit or null",
      "frequency":       "English or null",
      "duration":        "duration or null",
      "route":           "oral / nasal / IV / topical / etc or null",
      "notes":           "extra instructions or null",
      "read_confidence": "high" | "medium" | "low"
    }
  ],
  "raw_text":   "full verbatim text",
  "confidence": "high" | "medium" | "low"
}"""

USER_PROMPT = (
    "Read this prescription image carefully from TOP to BOTTOM. "
    "Extract EVERY medicine — do not stop early. "
    "Each R/ or Rx symbol = one medicine entry. "
    "Copy each drug name EXACTLY as handwritten. "
    "Return ONLY complete valid JSON."
)

# ══════════════════════════════════════════════════════════════════════════
# LOCAL CORRECTIONS DICTIONARY
# Applied AFTER OCR, BEFORE Gemini identification.
# Add known OCR misreads or doctor-specific misspellings here.
# Keys are case-insensitive.
# ══════════════════════════════════════════════════════════════════════════

_CORRECTIONS: dict[str, str] = {
    # ── OCR misreads (letter-shape confusion) ────────────────────────────
    "sulrenta":    "Sulperazone",   # S+ul+r+enta → Sulperazone (IV antibiotic)
    "alcamylac":   "Alcanyl",       # pancreatin brand
    "alamylase":   "Alcanyl",       # alternate OCR read of same drug
    "lactgemilan": "Lactulose",
    "lactsemilan": "Lactulose",
    "lactugemilan":"Lactulose",
    "vonapravin":  "Vonoprazan",    # vonoprazan (Takecab)
    "vonaprazin":  "Vonoprazan",
    "ciprafen":    "Ciprofloxacin", # ciprofloxacin (Ciprobay)
    "esomepraz":   "Esomeprazole",  # esomeprazole abbreviation
    "esomeprazol": "Esomeprazole",
    "gliben":      "Glibenclamide", # glibenclamide (Daonil)
    "glucophaj":   "Glucophage",
    "metformine":  "Metformin",
    "omeprazol":   "Omeprazole",
    "pantoprazol": "Pantoprazole",
    "rabeprazol":  "Rabeprazole",
    "augmentine":  "Augmentin",
    "amoxycillin": "Amoxicillin",
    "amoxycilin":  "Amoxicillin",
}

# Drugs that are ONLY available as IV/injection — flag if route=oral
_IV_ONLY_DRUGS = {
    "sulperazone", "ceftriaxone injection", "meropenem",
    "piperacillin tazobactam", "vancomycin injection",
    "imipenem", "ertapenem", "colistin injection",
}


def _apply_corrections(medicines: list[dict]) -> list[dict]:
    """
    Apply local corrections dictionary to medicine names.
    Also flags drugs prescribed as oral that are IV-only (clinical safety check).
    """
    for med in medicines:
        raw = (med.get("name") or "").strip()
        corrected = _CORRECTIONS.get(raw.lower())
        if corrected:
            print(f"  [Correction] '{raw}' → '{corrected}' (local dict)")
            med["corrected_from"] = raw
            med["name"] = corrected

        # Route conflict check — warn if IV-only drug is written with oral route
        name_lower = (med.get("name") or "").lower()
        route_lower = (med.get("route") or "").lower()
        if any(iv in name_lower for iv in _IV_ONLY_DRUGS):
            if route_lower in ("oral", "po", "by mouth", ""):
                existing_note = med.get("notes") or ""
                med["notes"] = (existing_note + " ⚠️ IV-only drug — verify route with prescriber").strip()
                print(f"  [⚠️  Route Alert] '{med['name']}' is IV-only but route='{route_lower or 'unspecified'}'")

    return medicines


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def read_image(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime = mime_map.get(path.suffix.lower(), "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, mime


def _fix_truncated(text: str) -> str:
    clean = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    clean = re.sub(r"```$", "", clean).strip()
    if len(re.findall(r'(?<!\\)"', clean)) % 2 != 0:
        clean += '"'
    clean += "]" * (clean.count("[") - clean.count("]"))
    clean += "}" * (clean.count("{") - clean.count("}"))
    return clean


def _parse_json(text: str) -> dict | None:
    clean = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    clean = re.sub(r"```$", "", clean).strip()
    for attempt in [clean, _fix_truncated(clean)]:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        for attempt in [m.group(), _fix_truncated(m.group())]:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
    return None


def _safe_fallback(reason: str) -> dict:
    print(f"[!] {reason}")
    return {
        "doctor": None, "patient": None, "date": None,
        "medicines": [], "raw_text": "", "confidence": "low",
    }


def _ocr_pass(data_url: str) -> dict | None:
    """Single OCR pass."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
        )
        choice = response.choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish and finish not in ("stop", "end_turn"):
            print(f"  [!] Model stopped early (finish_reason={finish})")
        return _parse_json(choice.message.content)
    except Exception as e:
        print(f"  [!] API error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════════

def _conf_rank(c: str | None) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get((c or "low").lower(), 1)


def _deduplicate(medicines: list[dict], threshold: int = 90) -> list[dict]:
    """
    Remove duplicate medicine entries (same name, fuzzy ≥ threshold).
    Keeps the higher-confidence entry.
    """
    kept: list[dict] = []
    for med in medicines:
        name = (med.get("name") or "").upper().strip()
        is_dup = False
        for i, existing in enumerate(kept):
            ex_name = (existing.get("name") or "").upper().strip()
            if fuzz.WRatio(name, ex_name) >= threshold:
                if _conf_rank(med.get("read_confidence")) > _conf_rank(existing.get("read_confidence")):
                    kept[i] = med
                is_dup = True
                break
        if not is_dup:
            kept.append(med)
    return kept


# ══════════════════════════════════════════════════════════════════════════
# UNION MERGE (used when running two passes)
# ══════════════════════════════════════════════════════════════════════════

def _union_merge(list1: list[dict], list2: list[dict]) -> list[dict]:
    """
    Take the UNION of two medicine lists — never lose an item.

    For each medicine in list2:
      - If it matches something in list1 (fuzzy ≥ 85) → keep the
        higher-confidence version (already in list1 via dedup)
      - If no match → add it (it was missed in list1)

    Result is then deduplicated.
    """
    merged = list(list1)  # start with all of list1
    for med2 in list2:
        name2 = (med2.get("name") or "").upper().strip()
        found = any(
            fuzz.WRatio(name2, (m.get("name") or "").upper().strip()) >= 85
            for m in merged
        )
        if not found:
            # list2 found a medicine that list1 missed — add it
            print(f"  [Merge] list2 found extra medicine: '{med2.get('name')}'")
            merged.append(med2)
    return _deduplicate(merged)


# ══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def analyze_prescription(image_path: str) -> dict:
    """
    OCR strategy:
      1. Run Pass 1 — single strong pass
      2. If result confidence is 'low' OR fewer than expected items
         (heuristic: < 3 items) → run Pass 2 and take UNION
      3. Deduplicate final list
    """
    try:
        image_data, mime_type = read_image(image_path)
    except Exception as e:
        return _safe_fallback(f"Could not read image: {e}")

    data_url = f"data:{mime_type};base64,{image_data}"

    # ── Pass 1 ────────────────────────────────────────────────────────────
    print("  [OCR] Pass 1...")
    result1 = _ocr_pass(data_url)
    if not result1:
        return _safe_fallback("OCR Pass 1 failed")

    meds1     = result1.get("medicines", [])
    meds1     = _apply_corrections(meds1)   # ← local dict fix before any Gemini call
    overall_c = (result1.get("confidence") or "medium").lower()

    # ── Pass 2: only if result looks incomplete ───────────────────────────
    # Heuristic: run a second pass if overall confidence is low
    # OR if we got very few medicines (likely missed some)
    run_pass2 = overall_c == "low"
    if run_pass2:
        print(f"  [OCR] Pass 1 returned {len(meds1)} medicine(s) "
              f"(conf={overall_c}) → running Pass 2 for completeness...")
        result2 = _ocr_pass(data_url)
        if result2:
            meds2 = result2.get("medicines", [])
            print(f"  [OCR] Pass 2 returned {len(meds2)} medicine(s)")
            # Union merge — never lose items
            meds1 = _union_merge(meds1, meds2)
            # Use better overall confidence
            c2 = (result2.get("confidence") or "medium").lower()
            conf_rank = {"high": 3, "medium": 2, "low": 1}
            overall_c = (
                overall_c if conf_rank.get(overall_c, 0) >= conf_rank.get(c2, 0)
                else c2
            )
        else:
            print("  [OCR] Pass 2 failed — keeping Pass 1 result")
    else:
        meds1 = _deduplicate(meds1)

    print(f"  [OCR] Final: {len(meds1)} medicine(s) extracted")

    return {
        "doctor":    result1.get("doctor"),
        "patient":   result1.get("patient"),
        "date":      result1.get("date"),
        "medicines": meds1,
        "raw_text":  result1.get("raw_text", ""),
        "confidence": overall_c,
    }


# ══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════

_CONF_ICON = {"high": "✅", "medium": "⚠️ ", "low": "❓"}

def print_results(result: dict):
    print("\n" + "=" * 58)
    print("   📋  PRESCRIPTION ANALYSIS")
    print("=" * 58)

    if result.get("doctor"):  print(f"  Doctor    : {result['doctor']}")
    if result.get("patient"): print(f"  Patient   : {result['patient']}")
    if result.get("date"):    print(f"  Date      : {result['date']}")

    conf_map = {"high": "✅ CLEAR", "medium": "⚠️  MODERATE", "low": "❌ UNCLEAR"}
    print(f"  Clarity   : {conf_map.get(result.get('confidence', ''), '')}")

    medicines = result.get("medicines", [])
    if medicines:
        print(f"\n  Medicines ({len(medicines)}):")
        print("  " + "─" * 52)
        for i, med in enumerate(medicines, 1):
            rc   = (med.get("read_confidence") or "high").lower()
            icon = _CONF_ICON.get(rc, "")
            print(f"\n  {i}. {icon} {med.get('name', '—')}")
            if med.get("dose"):      print(f"     Dose      : {med['dose']}")
            if med.get("frequency"): print(f"     Frequency : {med['frequency']}")
            if med.get("duration"):  print(f"     Duration  : {med['duration']}")
            if med.get("route"):     print(f"     Route     : {med['route']}")
            if med.get("notes"):     print(f"     Notes     : {med['notes']}")
    else:
        print("\n  No medicines detected.")

    if result.get("raw_text"):
        print("\n  Extracted Text:")
        print("  " + "─" * 52)
        for line in result["raw_text"].splitlines():
            print(f"  {line}")

    print("\n" + "=" * 58)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage  : python prescription_agent.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: '{image_path}' not found.")
        sys.exit(1)

    print(f"Analyzing: {image_path} ...")
    result = analyze_prescription(image_path)
    print_results(result)

    out = Path(image_path).stem + "_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved to : {out}\n")