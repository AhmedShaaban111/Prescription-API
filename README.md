# Prescription API

AI-powered handwritten prescription OCR and drug matching API, built with FastAPI + Gemini.

---

## Features

- **OCR**: Extracts medicine names, doses, frequency, route, and duration from prescription images
- **Drug matching**: Matches brand names, generics (INN), Arabic names, misspellings, and abbreviations against a pharmacy database
- **Dose-aware re-ranking**: Re-ranks results based on prescribed dose and route
- **Auto-correction**: Built-in dictionary for common OCR misreads

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp env.example .env
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
MODEL=gemini-2.0-flash
GROQ_API_KEY=your_groq_key_here          # optional
HUGGINGFACEHUB_API_TOKEN=your_hf_token   # optional
```

### 3. Run the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Reference

### `GET /health`
Returns service status.

```json
{ "status": "ok", "index_ready": true }
```

---

### `POST /analyze`
Upload a prescription image and get OCR + drug matching results.

**Parameters (query):**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `k` | int | 5 | Max DB matches per drug |
| `json_only` | bool | false | Skip terminal output |

**Body:** `multipart/form-data` — `file` field with image (JPEG, PNG, WEBP, GIF)

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/analyze?k=5" \
  -F "file=@prescription.jpg"
```

**Response:**
```json
{
  "generated_at": "2024-01-01T12:00:00",
  "image": "prescription.jpg",
  "ocr": {
    "doctor": "Dr. Smith",
    "patient": "John Doe",
    "date": "2024-01-01",
    "confidence": "high",
    "raw_text": "..."
  },
  "medicines": [
    {
      "ocr_name": "Nexium",
      "cleaned_name": "Nexium",
      "dose": "40mg",
      "frequency": "once daily",
      "duration": "30 days",
      "route": "oral",
      "notes": null,
      "read_confidence": "high",
      "corrected_from": null,
      "matching": {
        "input_type": "exact",
        "active": null,
        "category": null,
        "confidence": "EXACT",
        "note": null,
        "matches": ["NEXIUM 40MG CAPSULE", "NEXIUM 20MG CAPSULE"]
      }
    }
  ],
  "summary": { "total": 3, "found": 3, "not_found": 0 }
}
```

---

### `POST /match`
Match a list of drug names directly (no image needed).

**Body (JSON):**
```json
{
  "names": ["Nexium", "Glucophage 500", "Ciprafen"],
  "k": 5,
  "meta": {
    "Nexium": { "dose": "40mg", "frequency": "once daily", "route": "oral" }
  }
}
```

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/match" \
  -H "Content-Type: application/json" \
  -d '{"names": ["Nexium", "metformin", "امارايل"], "k": 3}'
```

**Response:**
```json
{
  "results": {
    "Nexium": {
      "input_type": "exact",
      "active": null,
      "category": null,
      "confidence": "EXACT",
      "note": null,
      "matches": ["NEXIUM 40MG CAPSULE", "NEXIUM 20MG CAPSULE"]
    }
  }
}
```

---

## Interactive Docs

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Project Structure

```
prescription_api/
├── main.py                    # FastAPI application
├── pipeline.py                # OCR + matching pipeline glue
├── prescription_agent.py      # Gemini OCR module
├── drug_pipeline.py           # Drug matching module
├── drug_bm25_index.pkl        # Pre-built BM25 search index
├── drug_faiss_index/          # FAISS vector index
│   ├── index.faiss
│   └── index.pkl
├── arabic_map.json            # Arabic drug name mappings
├── Book1.xlsx                 # Primary drug database
├── Druglist.csv               # Drug list (CSV)
├── DrugListLive_json.xlsx     # Drug list (live)
├── requirements.txt           # Python dependencies
└── env.example                # Environment variables template
```

---

## Disclaimer

> ⚠️ AI-assisted analysis — for research purposes only.  
> Always verify with a licensed pharmacist before dispensing.
