"""
drug_pipeline.py — Universal Pharmacy Item Matcher
===================================================

الهدف الأساسي: تعرف "الدواء ده إيه" — سواء كتب الدكتور:
  - اسم تجاري (brand):          Nexium, Gaviscon, Ciprobay
  - مادة فعالة (active/INN):    esomeprazole, ciprofloxacin, metformin
  - اسم مكتوب غلط / مختصر:     Ciprafen, Esomepraz, Alcamylac
  - بالعربي:                    امارايل, ديابيكون
  - مكمل غذائي / OTC:           Vitamin C 1000, Omega 3

Pipeline:
  Level 1 — Exact:  الاسم موجود مباشرة في الـ DB
  Level 2 — Fuzzy:  قريب من اسم في الـ DB (typo / abbreviation)
  Level 3 — Gemini: اسأل الـ LLM — ده brand ولا active ولا إيه؟
                    → لو active  : دور على brands في الـ DB
                    → لو brand   : دور عليه وعلى alternatives
                    في الحالتين : وضّح للمستخدم نوع الإدخال
"""

import os, re, json, pickle
from pathlib import Path
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from openai import OpenAI

load_dotenv()

INDEX_PATH      = "./drug_bm25_index.pkl"
ARABIC_RE       = re.compile(r'[\u0600-\u06FF]')
FUZZY_SCORE_MIN = 85
BRAND_CUTOFF    = 82

_IV_KW = {"INFUSION", "INJECTION", "INJECTABLE", "IV", "VIAL",
          "AMPOULE", "AMP", "IM", "SC", "INTRAVENOUS"}


# ══════════════════════════════════════════════════════════════════════════
# GEMINI CLIENT — module-level singleton
# ══════════════════════════════════════════════════════════════════════════

def _make_client() -> OpenAI | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

_CLIENT: OpenAI | None = _make_client()


# ══════════════════════════════════════════════════════════════════════════
# INDEX
# ══════════════════════════════════════════════════════════════════════════

_INDEX_CACHE: dict | None = None

def build_index(xlsx_path: str, force: bool = False):
    global _INDEX_CACHE
    if Path(INDEX_PATH).exists() and not force:
        return
    print(f"[Index] Building from '{xlsx_path}'...")
    df = pd.read_excel(xlsx_path)
    df["DRUGNAME"] = df["DRUGNAME"].astype(str).str.strip()
    df = df[df["DRUGNAME"] != ""].drop_duplicates()

    brand_map = defaultdict(list)
    for name in df["DRUGNAME"]:
        first = name.split()[0].upper()
        brand_map[first].append(name)

    docs      = [Document(page_content=n) for n in df["DRUGNAME"]]
    retriever = BM25Retriever.from_documents(docs)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({
            "retriever": retriever,
            "names":     df["DRUGNAME"].tolist(),
            "brand_map": dict(brand_map),
        }, f)
    _INDEX_CACHE = None
    print(f"[Index] {len(docs):,} items, {len(brand_map):,} unique brands → '{INDEX_PATH}'")


def _load() -> dict:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        with open(INDEX_PATH, "rb") as f:
            _INDEX_CACHE = pickle.load(f)
    return _INDEX_CACHE


# ══════════════════════════════════════════════════════════════════════════
# SEARCH HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_records(brand_map: dict, key: str, k: int) -> list[str]:
    return brand_map.get(key.upper(), [])[:k]


def _length_ok(query: str, brand: str) -> bool:
    q, b = len(query), len(brand)
    return (q / b) >= 0.50 if q <= b else (b / q) >= 0.72


def _fuzzy_brands(brand_map: dict, query: str, k: int, cutoff: int) -> list[tuple[str, float]]:
    hits = process.extract(
        query.upper(),
        list(brand_map.keys()),
        scorer=fuzz.WRatio,
        limit=20,
        score_cutoff=cutoff,
    )
    results, seen = [], set()
    for brand, score, _ in sorted(hits, key=lambda x: -x[1]):
        if not _length_ok(query.upper(), brand):
            continue
        for record in brand_map[brand]:
            if record not in seen:
                seen.add(record)
                results.append((record, score))
    return results[:k]


# ══════════════════════════════════════════════════════════════════════════
# GEMINI — classify + identify
# ══════════════════════════════════════════════════════════════════════════

_GEMINI_SYSTEM = """You are a clinical pharmacist expert in the Saudi Arabian (SFDA) pharmacy market.

Your job: given ANY pharmacy item name, identify it completely.

The name could be:
  - A brand/trade name:                 "Nexium", "Gaviscon", "Ciprobay"
  - An active ingredient (INN/generic): "esomeprazole", "metformin", "ciprofloxacin"
  - A misspelled brand or generic:      "Ciprafen" (→ ciprofloxacin), "Sulrenta" (→ vortioxetine)
  - Arabic brand or generic name:       "امارايل" (→ glimepiride)
  - Abbreviation or partial name:       "Esomepraz", "Alcamylac"
  - A supplement or OTC product:        "Vitamin C 1000", "Omega 3", "Liquid Chlorophyll"

Return ONLY valid JSON, no markdown:
{
  "input_type": "brand" | "active" | "misspelling" | "abbreviation" | "arabic" | "supplement" | "unknown",
  "active_ingredient": "the INN/generic name in English, lowercase",
  "category": "rx" | "otc" | "supplement" | "other",
  "brands": ["FIRST_WORD_UPPERCASE", ...],
  "confidence": "high" | "low"
}

Rules:
- input_type "active"  → doctor wrote the INN/generic name directly (e.g. "metformin", "esomeprazole")
- input_type "brand"   → doctor wrote a known trade name (e.g. "Nexium", "Gaviscon")
- input_type "misspelling" → written name is a misspelled brand or generic
- input_type "abbreviation" → shortened version of a known drug name
- brands = most common Saudi Arabia brand names for this active ingredient
- Return FIRST WORD ONLY of each brand, UPPERCASE
- If you are NOT 100% certain the drug exists → set confidence to "low" and brands to []
- Do NOT guess wildly — an uncertain answer is safer than a wrong one
- If the input could be a GI drug (given context like PPIs or antibiotics nearby),
  prioritize GI interpretations over psychiatric or unrelated drug classes

Examples:
"Nexium"            → {"input_type":"brand",        "active_ingredient":"esomeprazole",      "category":"rx",         "brands":["NEXIUM","ESOMEP","EMANERA"],         "confidence":"high"}
"esomeprazole"      → {"input_type":"active",       "active_ingredient":"esomeprazole",      "category":"rx",         "brands":["NEXIUM","ESOMEP","EMANERA"],         "confidence":"high"}
"Esomepraz"         → {"input_type":"abbreviation", "active_ingredient":"esomeprazole",      "category":"rx",         "brands":["NEXIUM","ESOMEP","EMANERA"],         "confidence":"high"}
"metformin"         → {"input_type":"active",       "active_ingredient":"metformin",         "category":"rx",         "brands":["GLUCOPHAGE","GLIFOR","METFORMIN"],   "confidence":"high"}
"Glucophage"        → {"input_type":"brand",        "active_ingredient":"metformin",         "category":"rx",         "brands":["GLUCOPHAGE","GLIFOR","METFORMIN"],   "confidence":"high"}
"Ciprafen"          → {"input_type":"misspelling",  "active_ingredient":"ciprofloxacin",     "category":"rx",         "brands":["CIPROBAY","CIPROFLOXACIN","CIPRO"],  "confidence":"high"}
"Sulrenta"          → {"input_type":"misspelling",  "active_ingredient":"vortioxetine",      "category":"rx",         "brands":["BRINTELLIX"],                       "confidence":"high"}
"Alcamylac"         → {"input_type":"misspelling",  "active_ingredient":"pancreatin",        "category":"rx",         "brands":["CREON","PANKREOFLAT"],              "confidence":"high"}
"Lactsemilan"       → {"input_type":"misspelling",  "active_ingredient":"lactulose",         "category":"rx",         "brands":["DUPHALAC","LACTULOSE"],             "confidence":"high"}
"Gaviscon"          → {"input_type":"brand",        "active_ingredient":"alginate antacid",  "category":"otc",        "brands":["GAVISCON"],                         "confidence":"high"}
"Omega 3"           → {"input_type":"supplement",   "active_ingredient":"omega-3 fatty acids","category":"supplement","brands":["OMACOR","OMEGA","MAXEPA"],          "confidence":"high"}
"Liquid Chlorophyll"→ {"input_type":"supplement",   "active_ingredient":"chlorophyll",       "category":"supplement", "brands":[],                                   "confidence":"low"}
"امارايل"           → {"input_type":"arabic",       "active_ingredient":"glimepiride",       "category":"rx",         "brands":["AMARYL","SOLOSA"],                  "confidence":"high"}
"Vonoprazan"        → {"input_type":"active",       "active_ingredient":"vonoprazan",        "category":"rx",         "brands":["VOQUEZNA","TAKECAB"],               "confidence":"high"}"""


def _gemini_lookup(item_name: str) -> dict:
    """
    Ask Gemini to classify and identify the pharmacy item.
    Returns: {input_type, active_ingredient, category, brands, confidence}
    """
    empty = {
        "input_type": "unknown", "active_ingredient": "",
        "category": "other", "brands": [], "confidence": "low",
    }
    if _CLIENT is None:
        return empty
    model = os.getenv("MODEL", "gemini-2.0-flash")
    try:
        response = _CLIENT.chat.completions.create(
            model=model,
            max_tokens=300,
            temperature=0,
            messages=[
                {"role": "system", "content": _GEMINI_SYSTEM},
                {"role": "user",   "content": f'Pharmacy item: "{item_name.title()}"'},
            ],
        )
        raw  = re.sub(r"```(?:json)?|```", "", response.choices[0].message.content.strip()).strip()
        data = json.loads(raw)
        return {
            "input_type":        str(data.get("input_type",        "unknown")).strip().lower(),
            "active_ingredient": str(data.get("active_ingredient", "")).strip().lower(),
            "category":          str(data.get("category",          "other")).strip().lower(),
            "brands":            [b.strip().split()[0].upper()
                                  for b in data.get("brands", []) if isinstance(b, str)],
            "confidence":        str(data.get("confidence",        "low")).strip().lower(),
        }
    except Exception as e:
        print(f"  [Gemini] Error: {e}")
        return empty


# ══════════════════════════════════════════════════════════════════════════
# DOSE-AWARE RE-RANKING
# ══════════════════════════════════════════════════════════════════════════

def _closest_dose_score(nums_in_name: list[str], dose_num: str) -> int:
    """Score a product by how close its strength is to the prescribed dose."""
    if not dose_num or not nums_in_name:
        return 0
    target = int(dose_num)
    valid  = [int(n) for n in nums_in_name if n.isdigit() and 0 < int(n) < 10000]
    if not valid:
        return 0
    diff = min(abs(n - target) for n in valid)
    if diff == 0:   return +15
    if diff <= 5:   return +10
    if diff <= 15:  return +6
    if diff <= 50:  return +2
    return -5


def _rerank(matches: list[str], route: str | None, dose: str | None) -> list[str]:
    """
    Re-rank matches:
      - Penalise IV/INJECTION when route=oral (-30)
      - Prefer closest dose strength (no flat oral bonus — avoids
        arbitrary reordering when all candidates are oral forms)
    Uses stable sort so equal-score items keep original order.
    """
    if not matches:
        return matches
    dose_num = re.sub(r'[^\d]', '', str(dose)) if dose else ""

    def _score(name: str) -> int:
        upper = name.upper()
        s = 0
        if route and route.lower() in ("oral", "po", "by mouth"):
            if set(upper.split()) & _IV_KW:
                s -= 30
        s += _closest_dose_score(re.findall(r'\d+', upper), dose_num)
        return s

    return sorted(matches, key=_score, reverse=True)


# ══════════════════════════════════════════════════════════════════════════
# CORE MATCH
# ══════════════════════════════════════════════════════════════════════════

def match(
    item_name: str,
    k:     int = 5,
    route: str | None = None,
    dose:  str | None = None,
) -> dict:
    """
    Identify and match any pharmacy item against the DB.

    Handles: brand names, active ingredients (INN), misspellings,
             Arabic names, abbreviations, supplements, OTC.

    Returns:
    {
        "input_type":  "exact" | "fuzzy" | "brand" | "active" |
                       "misspelling" | "abbreviation" | "arabic" |
                       "supplement" | "unknown",
        "active":      str | None,    ← INN/generic name from Gemini
        "category":    str | None,    ← rx / otc / supplement / other
        "matches":     [str],         ← DB product names, best first
        "confidence":  "EXACT" | "HIGH" | "MEDIUM" | "LOW" | "NOT_FOUND",
        "note":        str | None,
    }
    """
    data      = _load()
    brand_map = data["brand_map"]
    query_up  = item_name.strip().upper()
    is_arabic = bool(ARABIC_RE.search(item_name))

    # ── Level 1: Exact key in DB ──────────────────────────────────────────
    if not is_arabic:
        records = _get_records(brand_map, query_up, k)
        if records:
            return {
                "input_type": "exact",
                "active":     None,
                "category":   None,
                "matches":    _rerank(records, route, dose),
                "confidence": "EXACT",
                "note":       None,
            }

    # ── Level 2: Fuzzy typo / abbreviation ───────────────────────────────
    if not is_arabic:
        hits = _fuzzy_brands(brand_map, query_up, k, cutoff=FUZZY_SCORE_MIN)
        if hits:
            return {
                "input_type": "fuzzy",
                "active":     None,
                "category":   None,
                "matches":    _rerank([n for n, _ in hits], route, dose),
                "confidence": "HIGH",
                "note":       None,
            }

    # ── Level 3: Gemini — classify + resolve ─────────────────────────────
    print(f"  [Gemini] '{item_name}' → identifying...")
    g        = _gemini_lookup(item_name)
    active   = g["active_ingredient"]
    category = g["category"]
    brands   = g["brands"]
    gconf    = g["confidence"]
    itype    = g["input_type"]
    print(f"  [Gemini] Type={itype} | Active='{active}' | Brands={brands} | Conf={gconf}")

    # Completely unidentifiable
    if not brands and not active:
        return {
            "input_type": itype,
            "active":     None,
            "category":   category,
            "matches":    [],
            "confidence": "NOT_FOUND",
            "note":       None,
        }

    # Search DB: try every brand hint + active ingredient
    seen: dict[str, float] = {}
    for term in brands + ([active] if active else []):
        for rec in _get_records(brand_map, term, k):
            seen[rec] = max(seen.get(rec, 0), 100.0)
        for rec, score in _fuzzy_brands(brand_map, term, k, cutoff=BRAND_CUTOFF):
            seen[rec] = max(seen.get(rec, 0), score)

    # Build human-readable note based on input_type
    if itype == "active":
        note = f"⚗️  Written as active ingredient — showing available brands"
    elif itype in ("misspelling", "abbreviation"):
        note = f"Active: {active}" if active else None
    elif itype == "arabic":
        note = f"Active: {active}" if active else None
    else:
        note = f"Active: {active}" if active else None

    if not seen:
        if active:
            note = (f"⚗️  Active ingredient '{active}' ({category}) — not found in database"
                    if itype == "active"
                    else f"Identified as '{active}' ({category}) — not found in database")
        return {
            "input_type": itype,
            "active":     active or None,
            "category":   category,
            "matches":    [],
            "confidence": "NOT_FOUND",
            "note":       note,
        }

    pre_ranked = [n for n, _ in sorted(seen.items(), key=lambda x: -x[1])[:k]]
    ranked     = _rerank(pre_ranked, route, dose)
    conf       = "MEDIUM" if gconf == "high" else "LOW"

    return {
        "input_type": itype,
        "active":     active or None,
        "category":   category,
        "matches":    ranked,
        "confidence": conf,
        "note":       note,
    }


def match_many(
    item_names: list[str],
    k:    int = 5,
    meta: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """
    Match a list of drug names.
    meta: {name: {route, dose}} for re-ranking.
    """
    meta = meta or {}
    return {
        name: match(
            name, k,
            route=meta.get(name, {}).get("route"),
            dose=meta.get(name,  {}).get("dose"),
        )
        for name in item_names if name.strip()
    }


# ══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════

_CATEGORY_BADGE = {
    "rx":         "💊 Rx",
    "otc":        "🏪 OTC",
    "supplement": "🌿 Supp",
    "other":      "📦 Other",
}

# Human-readable label per input_type
_ITYPE_LABEL = {
    "exact":        "✓✓ Exact match",
    "fuzzy":        "✓  Fuzzy match",
    "brand":        "~  Brand (Gemini)",
    "active":       "⚗️  Active ingredient",
    "misspelling":  "~  Misspelling resolved",
    "abbreviation": "~  Abbreviation resolved",
    "arabic":       "~  Arabic resolved",
    "supplement":   "~  Supplement",
    "unknown":      "⚠  Uncertain",
}

_CONF_COLOR = {
    "EXACT":     "🟢",
    "HIGH":      "🟢",
    "MEDIUM":    "🟡",
    "LOW":       "🟠",
    "NOT_FOUND": "🔴",
}


def print_results(
    results:           dict[str, dict],
    prescription_meta: dict[str, dict] | None = None,
):
    """
    Print enriched matching results.
    prescription_meta: {drug_name: {dose, frequency, duration, route, notes}}
    """
    W    = 72
    meta = prescription_meta or {}

    found    = sum(1 for v in results.values() if v["confidence"] != "NOT_FOUND")
    notfound = len(results) - found

    print()
    print("┌" + "─" * (W - 2) + "┐")
    print("│" + "  📋  PRESCRIPTION MATCHING RESULTS".center(W - 2) + "│")
    print("├" + "─" * (W - 2) + "┤")

    for idx, (query, result) in enumerate(results.items(), 1):
        conf      = result["confidence"]
        color     = _CONF_COLOR[conf]
        matches   = result["matches"]
        category  = result.get("category")
        active    = result.get("active")
        note      = result.get("note")
        itype     = result.get("input_type", "")
        badge     = _CATEGORY_BADGE.get(category, "") if category else ""
        drug_meta = meta.get(query, {})
        label     = _ITYPE_LABEL.get(itype, "~  Gemini")

        print("│" + " " * (W - 2) + "│")

        # ── Drug name + badge ─────────────────────────────────────────────
        header  = f"  {color} {idx}. {query.upper()}"
        padding = W - 2 - len(header) - len(badge)
        print("│" + header + " " * max(padding, 1) + badge + "│")

        # ── Rx info line ──────────────────────────────────────────────────
        rx_parts = []
        if drug_meta.get("dose"):      rx_parts.append(f"Dose: {drug_meta['dose']}")
        if drug_meta.get("frequency"): rx_parts.append(f"Freq: {drug_meta['frequency']}")
        if drug_meta.get("duration"):  rx_parts.append(f"Dur:  {drug_meta['duration']}")
        if drug_meta.get("route"):     rx_parts.append(f"Route: {drug_meta['route']}")
        if rx_parts:
            print("│" + ("     " + "  │  ".join(rx_parts)).ljust(W - 2) + "│")
        if drug_meta.get("notes"):
            print("│" + f"     📝 {drug_meta['notes']}".ljust(W - 2) + "│")

        # ── Match type + resolved active ingredient ───────────────────────
        sub = f"     {label}"
        if active and itype not in ("exact", "fuzzy"):
            sub += f"  →  {active}"
        print("│" + sub.ljust(W - 2) + "│")

        # ── Divider ───────────────────────────────────────────────────────
        print("│" + "     " + "·" * (W - 7) + "│")

        # ── DB matches ────────────────────────────────────────────────────
        if conf == "NOT_FOUND":
            print("│" + f"     ✗  {note or 'Not found in database'}".ljust(W - 2) + "│")
        else:
            for i, m in enumerate(matches, 1):
                prefix = "  ★ " if i == 1 else f"  {i}."
                print("│" + f"     {prefix} {m}".ljust(W - 2) + "│")

    # ── Footer ────────────────────────────────────────────────────────────
    print("├" + "─" * (W - 2) + "┤")
    print("│" + f"  Total: {len(results)}   🟢 Found: {found}   🔴 Not Found: {notfound}".ljust(W - 2) + "│")
    print("│" + "  ✓✓ Exact  ✓ Fuzzy  ⚗️ Active  ~ Gemini  ⚠ Uncertain  ✗ Not found".ljust(W - 2) + "│")
    print("├" + "─" * (W - 2) + "┤")
    print("│" + "  ⚠️  DISCLAIMER: AI-assisted analysis — for research purposes only.".ljust(W - 2) + "│")
    print("│" + "     Always verify with a licensed pharmacist before dispensing.".ljust(W - 2) + "│")
    print("└" + "─" * (W - 2) + "┘")
    print()