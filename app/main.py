"""FastAPI service: upload a document -> async OCR + extraction -> poll for result.

Flow:  POST /extract  ->  Redis queue  ->  Celery worker  ->  GET /result/{id}

Extra endpoint:
  POST /preview  -- returns the preprocessed image as a PNG; useful for
                    debugging whether the cleaning pipeline improved the input.
"""
import io
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .celery_app import celery_app
from .image_processing import load_image_from_bytes, preprocess_for_ocr
from .worker import process_document

_STATIC_DIR = Path(__file__).parent / "static"

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Document Field Extractor",
    description="Async OCR + structured field extraction for invoices and receipts.",
    version="2.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

ALLOWED_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/jpg"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "document-field-extractor",
        "version": app.version,
    }


@app.post("/extract", status_code=202)
@limiter.limit("10/minute")
async def extract(request: Request, file: UploadFile = File(...)):
    """Accept a document and queue OCR + extraction in the background.

    Returns a job_id. Poll GET /result/{job_id} for the outcome.
    """
    _validate_upload(file)
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    job_id = str(uuid.uuid4())
    process_document.apply_async(
        args=[contents, file.filename, file.content_type],
        task_id=job_id,
    )
    return {"job_id": job_id, "status": "queued"}


@app.post("/preview")
@limiter.limit("20/minute")
async def preview(request: Request, file: UploadFile = File(...)):
    """Return the image after preprocessing as a PNG.

    Lets you see exactly what Tesseract receives -- handy for diagnosing
    poor extraction results without having to rebuild the whole pipeline.
    PDFs are not supported here (use /extract for those).
    """
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(
            status_code=415,
            detail="Preview only supports PNG/JPEG. Use /extract for PDFs.",
        )
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    pil_in = load_image_from_bytes(contents)
    pil_out = preprocess_for_ocr(pil_in)

    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/result/{job_id}")
def result(job_id: str):
    """Poll for a queued job.

    States: queued/processing (still running), done (data ready),
    failed (surface the error message).
    """
    res = celery_app.AsyncResult(job_id)

    if res.state == "PENDING":
        return {"job_id": job_id, "status": "processing"}
    if res.state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(res.result)}
    if res.state == "SUCCESS":
        return {"job_id": job_id, "status": "done", "result": res.result}

    return {"job_id": job_id, "status": res.state.lower()}


# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type '{file.content_type}'. Send a PDF, PNG, or JPEG.",
        )
