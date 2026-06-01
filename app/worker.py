"""Background worker: OCR + field extraction, run inside Celery.

Two stages:
  1. OCR        -- raw bytes -> text (via Tesseract, optionally via pdf2image)
  2. Extraction -- text -> structured fields

Image preprocessing (deskew, denoise, threshold) happens between the two:
it converts whatever the user uploaded into the clean binary form that
Tesseract handles best. That step lives in image_processing.py.
"""
import io

from .celery_app import celery_app
from .extraction import extract_fields
from .image_processing import load_image_from_bytes, preprocess_for_ocr


def _run_tesseract(pil_image) -> str:
    import pytesseract
    import os
    if os.name == "nt":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    cleaned = preprocess_for_ocr(pil_image)
    return pytesseract.image_to_string(cleaned, config="--oem 3 --psm 6")


def _ocr_image(image_bytes: bytes) -> str:
    img = load_image_from_bytes(image_bytes)
    return _run_tesseract(img)


def _ocr_pdf(pdf_bytes: bytes) -> str:
    """Render each PDF page to an image, preprocess, then OCR."""
    from pdf2image import convert_from_bytes

    pages = convert_from_bytes(pdf_bytes)
    return "\n".join(_run_tesseract(page) for page in pages)


@celery_app.task(name="process_document", bind=True)
def process_document(self, contents: bytes, filename: str, content_type: str):
    if content_type == "application/pdf":
        raw_text = _ocr_pdf(contents)
    else:
        raw_text = _ocr_image(contents)

    fields = extract_fields(raw_text)

    return {
        "filename": filename,
        "fields": fields,
        "raw_text_preview": raw_text[:500],
    }
