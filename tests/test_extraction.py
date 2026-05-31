"""Tests for the extraction layer. These run without OCR or Redis --
they exercise the logic that matters most, on text directly."""
from app.extraction import extract_fields, REVIEW_THRESHOLD


def test_clean_invoice_extracts_total_not_subtotal():
    text = "Subtotal: $450.00\nTax: $45.00\nTotal: $495.00"
    fields = extract_fields(text)["fields"]
    assert fields["total"]["value"] == "495.00"
    assert fields["total"]["needs_review"] is False


def test_invoice_number_not_truncated():
    text = "Invoice No: INV-2024-0042"
    assert extract_fields(text)["fields"]["invoice_number"]["value"] == "INV-2024-0042"


def test_missing_fields_are_flagged_for_review():
    text = "just some text with no useful fields"
    result = extract_fields(text)
    assert "total" in result["needs_review"]
    assert result["fields"]["total"]["value"] is None


def test_email_extraction():
    text = "Contact us at billing@example.com for questions"
    assert extract_fields(text)["fields"]["email"]["value"] == "billing@example.com"


def test_low_confidence_is_flagged():
    # No labelled total, only a bare amount -> low confidence -> review.
    text = "Some receipt\n$12.99"
    total = extract_fields(text)["fields"]["total"]
    if total["value"] is not None:
        assert total["confidence"] <= REVIEW_THRESHOLD
        assert total["needs_review"] is True
