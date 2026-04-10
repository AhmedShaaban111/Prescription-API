"""
main.py — Prescription API
==========================
FastAPI wrapper around the prescription OCR + drug matching pipeline.

Endpoints:
  POST /analyze          — Upload a prescription image → OCR + drug matching
  POST /match            — Match a list of drug names against the DB
  GET  /health           — Health check
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline import run
from drug_pipeline import build_index, match_many

# ── Config ────────────────────────────────────────────────────────────────
XLSX_PATH = os.getenv("XLSX_PATH", "Book1.xlsx")
DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))

app = FastAPI(
    title="Prescription API",
    description=(
        "AI-powered handwritten prescription OCR and drug matching.\n\n"
        "- **`/analyze`** — Upload a prescription image to extract medicines and match them against the pharmacy DB.\n"
        "- **`/match`** — Match a list of drug names directly (no image needed).\n"
        "- **`/health`** — Service health check.\n\n"
        "> ⚠️ For research purposes only. Always verify with a licensed pharmacist before dispensing."
    ),
    version="1.0.0",
)


# ── Startup: pre-load DB index ────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    if Path(XLSX_PATH).exists():
        build_index(XLSX_PATH, force=False)
    else:
        print(f"[Warning] XLSX not found at '{XLSX_PATH}' — index not pre-built.")


# ══════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════

class MatchRequest(BaseModel):
    names: list[str]
    k: Optional[int] = DEFAULT_K
    meta: Optional[dict[str, dict]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "names": ["Nexium", "Glucophage 500", "Ciprafen"],
                "k": 5,
                "meta": {
                    "Nexium": {"dose": "40mg", "frequency": "once daily", "route": "oral"}
                },
            }
        }
    }


class DrugMatch(BaseModel):
    input_type: Optional[str]
    active: Optional[str]
    category: Optional[str]
    confidence: Optional[str]
    note: Optional[str]
    matches: list[str]


class MatchResponse(BaseModel):
    results: dict[str, DrugMatch]


class OCRResult(BaseModel):
    doctor: Optional[str]
    patient: Optional[str]
    date: Optional[str]
    confidence: Optional[str]
    raw_text: Optional[str]


class MedicineEntry(BaseModel):
    ocr_name: Optional[str]
    cleaned_name: Optional[str]
    dose: Optional[str]
    frequency: Optional[str]
    duration: Optional[str]
    route: Optional[str]
    notes: Optional[str]
    read_confidence: Optional[str]
    corrected_from: Optional[str]
    matching: Optional[DrugMatch]


class AnalyzeResponse(BaseModel):
    generated_at: str
    image: str
    ocr: OCRResult
    medicines: list[MedicineEntry]
    summary: dict


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Utility"])
def health():
    """Returns service status and whether the drug index is loaded."""
    index_ready = Path("drug_bm25_index.pkl").exists()
    return {"status": "ok", "index_ready": index_ready}


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Prescription"],
    summary="Analyze a prescription image",
    description=(
        "Upload a handwritten prescription image (JPEG, PNG, WEBP, GIF).\n\n"
        "Returns OCR-extracted medicine names with dose/frequency/route, "
        "matched against the pharmacy database.\n\n"
        "Supported image formats: `image/jpeg`, `image/png`, `image/webp`, `image/gif`."
    ),
)
async def analyze_prescription_endpoint(
    file: UploadFile = File(..., description="Prescription image file"),
    k: int = Query(DEFAULT_K, ge=1, le=20, description="Max DB matches per drug"),
    json_only: bool = Query(False, description="Skip terminal table output"),
    xlsx: str = Query(XLSX_PATH, description="Path to drug database XLSX file"),
):
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Must be one of: {sorted(allowed_types)}",
        )

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = run(tmp_path, xlsx_path=xlsx, k=k, json_only=json_only)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        # Clean up the result JSON file written by pipeline.run()
        result_json = Path(Path(tmp_path).stem + "_result.json")
        result_json.unlink(missing_ok=True)

    if result is None:
        raise HTTPException(status_code=422, detail="No medicines detected in image.")

    return JSONResponse(content=result)


@app.post(
    "/match",
    response_model=MatchResponse,
    tags=["Drug Matching"],
    summary="Match drug names against the pharmacy database",
    description=(
        "Match a list of drug names (brand, generic/INN, Arabic, misspellings) "
        "against the pharmacy database.\n\n"
        "Optionally provide per-drug metadata (`dose`, `frequency`, `route`) "
        "for dose-aware re-ranking."
    ),
)
def match_drugs(body: MatchRequest):
    if not body.names:
        raise HTTPException(status_code=400, detail="'names' list must not be empty.")

    try:
        raw = match_many(body.names, k=body.k or DEFAULT_K, meta=body.meta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"results": raw}
