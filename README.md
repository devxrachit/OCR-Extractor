# Document Field Extractor

An asynchronous OCR pipeline that extracts structured fields — vendor, invoice number, date, total, and email — from invoices and receipts. Every extracted field carries a **confidence score**, and low-confidence fields are automatically flagged for human review.

Built with **FastAPI · Celery · Redis · Tesseract · OpenCV · Docker**.

---

## Architecture

OCR on a real document can take several seconds. Blocking an HTTP request that long is not acceptable, so all heavy work runs off the request path:

```
        upload                      queue                     process
  client ──────▶ FastAPI ───────▶ Redis ───────▶ Celery worker
                    ▲                                   │
                    └─────── poll /result/{id} ◀────────┘
                                (result stored in Redis)
```

| Service | Role |
|---------|------|
| **FastAPI** | Accepts uploads, validates files, returns `job_id` immediately (HTTP 202) |
| **Redis** | Message broker + result store |
| **Celery worker** | Preprocesses image, runs Tesseract OCR, extracts fields |

The client polls `GET /result/{job_id}` until the job is done. This decoupling means the API stays responsive under load regardless of how long OCR takes.

---

## Image Preprocessing Pipeline

Raw images from phones and scanners are skewed, noisy, and low-contrast. Before Tesseract sees a single pixel, every image passes through a 5-step cleaning pipeline:

| Step | Implementation | Why |
|------|---------------|-----|
| Grayscale | `cv2.cvtColor` | Tesseract operates on intensity only |
| Scale up | `cv2.resize` (if height < 1000 px) | Accuracy drops sharply below ~150 dpi |
| Denoise | `cv2.fastNlMeansDenoising` | Removes salt-and-pepper noise from cheap scanners |
| Deskew | `cv2.minAreaRect` + rotate | Corrects tilt from uneven placement on a scanner bed |
| Adaptive threshold | `cv2.adaptiveThreshold` | Handles shadows and uneven lighting from phone photos |

Use `POST /preview` to see the preprocessed image before running the full pipeline — useful for diagnosing poor extraction results.

---

## Field Extraction

Turning raw OCR text into reliable structured data across inconsistent document layouts is the core challenge. This project solves it with a rule + regex approach where:

- Every extractor returns a **(value, confidence)** pair — no silent guesses.
- Fields below the confidence threshold are added to a `needs_review` list.

Example response:

```json
{
  "job_id": "abc-123",
  "status": "done",
  "result": {
    "filename": "invoice.png",
    "fields": {
      "fields": {
        "vendor":         { "value": "ACME SUPPLIES LTD", "confidence": 0.50, "needs_review": true  },
        "invoice_number": { "value": "INV-2024-0042",     "confidence": 0.85, "needs_review": false },
        "date":           { "value": "2024-01-31",        "confidence": 0.90, "needs_review": false },
        "total":          { "value": "495.00",            "confidence": 0.95, "needs_review": false },
        "email":          { "value": "billing@acme.com",  "confidence": 0.95, "needs_review": false }
      },
      "needs_review": ["vendor"],
      "summary": "4 of 5 fields extracted confidently."
    },
    "raw_text_preview": "ACME SUPPLIES LTD\n..."
  }
}
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/extract` | Upload a document — returns `job_id` immediately |
| `GET` | `/result/{job_id}` | Poll for the extraction result |
| `POST` | `/preview` | Return the preprocessed image as PNG |

**Supported formats:** PDF, PNG, JPEG

**Rate limits:** `/extract` — 10 requests/min per IP · `/preview` — 20 requests/min per IP

Exceeding the limit returns `429 Too Many Requests`.

Interactive docs available at `http://localhost:8000/docs` when running locally.

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Run

```bash
git clone https://github.com/YOUR_USERNAME/ocr-extractor.git
cd ocr-extractor
docker compose up --build
```

All three services (API, worker, Redis) start automatically.

### Usage

```bash
# Submit a document
curl -X POST http://localhost:8000/extract -F "file=@invoice.png"
# -> {"job_id": "abc-123", "status": "queued"}

# Poll for the result
curl http://localhost:8000/result/abc-123

# Preview the preprocessed image
curl -X POST http://localhost:8000/preview -F "file=@invoice.png" --output preview.png
```

### Stop

```bash
docker compose down
```

---

## Running Tests

Tests cover extraction logic and the image preprocessing pipeline independently — no OCR engine, Redis, or Docker required.

```bash
pip install -r requirements.txt
pytest tests/ -v
```

```
22 passed in 1.14s
```

---

## Project Structure

```
.
├── app/
│   ├── main.py            # FastAPI routes + rate limiting
│   ├── worker.py          # Celery task: OCR + extraction
│   ├── extraction.py      # Field extractors with confidence scores
│   ├── image_processing.py# 5-step preprocessing pipeline
│   └── celery_app.py      # Celery + Redis configuration
├── tests/
│   ├── test_extraction.py
│   └── test_image_processing.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Limitations & What's Next

- **Rule-based extraction** is fast and transparent but will miss unusual document layouts. A layout-aware model (e.g. LayoutLM) is the natural next step for fields where regex plateaus.
- **No persistence** beyond Redis result expiry (1 hour). A production deployment would store results in a database.
- **Vendor detection** is heuristic (first non-empty line) and intentionally carries low confidence so it always gets flagged for review.
