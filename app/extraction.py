"""Structured field extraction from raw OCR text.

This is the heart of the project. The approach is rule + regex based,
which is honest for a few-day build and genuinely how a lot of
production extractors start before ML is layered on.

Two ideas worth calling out in an interview:
  - Every field carries a CONFIDENCE. We never pretend a guess is a fact.
  - Low-confidence fields are flagged for review rather than trusted
    blindly -- which is exactly how real document systems behave.
"""
import re
from typing import Optional


# --- individual field extractors -------------------------------------------
# Each returns (value, confidence) so the caller can decide what to trust.

def _find_total(text: str) -> tuple[Optional[str], float]:
    """Grab the invoice/receipt total. We prefer a line that actually
    says 'total', and fall back to the largest currency amount on the
    page -- with lower confidence, since that's a guess."""
    # High confidence: an explicit "total ... <amount>" line. The leading
    # (?<![a-z]) stops us matching the "total" inside "Subtotal". We take
    # the LAST such match, since the final total is what we want when both
    # a subtotal-style line and a grand total appear.
    labelled = re.findall(
        r"(?<![a-z])(?:grand\s+total|total\s+due|total)\s*[:\-]?\s*[$€£]?\s*([\d,]+\.\d{2})",
        text,
        re.IGNORECASE,
    )
    if labelled:
        return labelled[-1].replace(",", ""), 0.95

    # Fallback: biggest currency-looking number anywhere.
    amounts = re.findall(r"[$€£]\s*([\d,]+\.\d{2})", text)
    if amounts:
        biggest = max(amounts, key=lambda a: float(a.replace(",", "")))
        return biggest.replace(",", ""), 0.55

    return None, 0.0


def _find_date(text: str) -> tuple[Optional[str], float]:
    """Find the first plausible date. Handles a few common formats."""
    patterns = [
        (r"\b(\d{4}-\d{2}-\d{2})\b", 0.9),                     # 2024-01-31
        (r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", 0.8),               # 31/01/2024
        (r"\b(\d{1,2}\s+\w+\s+\d{4})\b", 0.75),                # 31 Jan 2024
        (r"\b(\w+\s+\d{1,2},?\s+\d{4})\b", 0.75),              # Jan 31, 2024
    ]
    for pattern, conf in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1), conf
    return None, 0.0


def _find_invoice_number(text: str) -> tuple[Optional[str], float]:
    m = re.search(
        r"(?:invoice|inv|bill)\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]{2,})",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1), 0.85
    return None, 0.0


def _find_vendor(text: str) -> tuple[Optional[str], float]:
    """Heuristic: the first non-empty line is very often the vendor /
    business name on receipts and invoices. Low-ish confidence by nature."""
    for line in text.splitlines():
        clean = line.strip()
        if len(clean) >= 3 and not clean.isdigit():
            return clean, 0.5
    return None, 0.0


def _find_email(text: str) -> tuple[Optional[str], float]:
    m = re.search(r"[\w.\-]+@[\w.\-]+\.\w+", text)
    if m:
        return m.group(0), 0.95
    return None, 0.0


# --- orchestration ----------------------------------------------------------

# Anything at or below this confidence gets flagged for a human to check.
REVIEW_THRESHOLD = 0.6


def extract_fields(raw_text: str) -> dict:
    """Run every extractor and assemble a structured, self-describing
    result. Each field reports its value, a confidence score, and
    whether it should be reviewed."""
    extractors = {
        "vendor": _find_vendor,
        "invoice_number": _find_invoice_number,
        "date": _find_date,
        "total": _find_total,
        "email": _find_email,
    }

    fields = {}
    needs_review = []
    for name, fn in extractors.items():
        value, confidence = fn(raw_text)
        flagged = value is None or confidence <= REVIEW_THRESHOLD
        fields[name] = {
            "value": value,
            "confidence": round(confidence, 2),
            "needs_review": flagged,
        }
        if flagged:
            needs_review.append(name)

    return {
        "fields": fields,
        "needs_review": needs_review,
        "summary": (
            f"{len(extractors) - len(needs_review)} of {len(extractors)} "
            f"fields extracted confidently."
        ),
    }
