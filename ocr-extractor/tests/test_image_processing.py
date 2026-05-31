"""Tests for the image preprocessing pipeline.

These tests don't need Tesseract, Redis, or Docker -- they operate on
synthetic images constructed with numpy so they run anywhere.
"""
import io

import numpy as np
import pytest
from PIL import Image

from app.image_processing import (
    _adaptive_threshold,
    _deskew,
    _denoise,
    _scale_if_small,
    _to_gray,
    load_image_from_bytes,
    preprocess_for_ocr,
)


def _make_gray_pil(h: int = 200, w: int = 300, fill: int = 200) -> Image.Image:
    arr = np.full((h, w, 3), fill, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_noisy_pil(h: int = 200, w: int = 300) -> Image.Image:
    rng = np.random.default_rng(42)
    arr = np.full((h, w, 3), 200, dtype=np.uint8)
    noise_mask = rng.random((h, w)) < 0.05
    arr[noise_mask] = 0
    return Image.fromarray(arr, "RGB")


class TestToGray:
    def test_rgb_becomes_2d(self):
        pil = _make_gray_pil()
        arr = _to_gray(pil)
        assert arr.ndim == 2

    def test_output_dtype_uint8(self):
        arr = _to_gray(_make_gray_pil())
        assert arr.dtype == np.uint8


class TestScaleIfSmall:
    def test_small_image_is_scaled_up(self):
        arr = np.zeros((400, 300), dtype=np.uint8)
        out = _scale_if_small(arr)
        assert out.shape[0] > arr.shape[0]

    def test_large_image_is_unchanged(self):
        arr = np.zeros((1200, 900), dtype=np.uint8)
        out = _scale_if_small(arr)
        assert out.shape == arr.shape


class TestDenoise:
    def test_shape_preserved(self):
        arr = _to_gray(_make_noisy_pil())
        out = _denoise(arr)
        assert out.shape == arr.shape

    def test_output_dtype_uint8(self):
        arr = _to_gray(_make_noisy_pil())
        assert _denoise(arr).dtype == np.uint8


class TestDeskew:
    def test_straight_image_unchanged(self):
        arr = np.full((500, 400), 200, dtype=np.uint8)
        out = _deskew(arr)
        assert out.shape == arr.shape

    def test_sparse_image_returned_as_is(self):
        arr = np.full((200, 150), 255, dtype=np.uint8)
        out = _deskew(arr)
        assert out.shape == arr.shape

    def test_skewed_image_is_corrected(self):
        base = np.full((600, 800), 255, dtype=np.uint8)
        # Draw a horizontal line at y=300 -- then rotate it 5 degrees.
        import cv2
        cv2.line(base, (50, 300), (750, 300), 0, 4)
        M = cv2.getRotationMatrix2D((400, 300), 5, 1.0)
        skewed = cv2.warpAffine(base, M, (800, 600), borderValue=255)
        corrected = _deskew(skewed)
        assert corrected.shape == skewed.shape


class TestAdaptiveThreshold:
    def test_output_is_binary(self):
        arr = np.full((200, 300), 150, dtype=np.uint8)
        out = _adaptive_threshold(arr)
        unique = set(np.unique(out).tolist())
        assert unique.issubset({0, 255})

    def test_shape_preserved(self):
        arr = np.full((200, 300), 150, dtype=np.uint8)
        out = _adaptive_threshold(arr)
        assert out.shape == arr.shape


class TestPreprocessForOcr:
    def test_returns_pil_image(self):
        pil = _make_gray_pil()
        out = preprocess_for_ocr(pil)
        assert isinstance(out, Image.Image)

    def test_output_is_binary_mode(self):
        pil = _make_gray_pil()
        out = preprocess_for_ocr(pil)
        arr = np.array(out)
        unique = set(np.unique(arr).tolist())
        assert unique.issubset({0, 255})

    def test_handles_small_image(self):
        small = _make_gray_pil(h=100, w=150)
        out = preprocess_for_ocr(small)
        assert isinstance(out, Image.Image)

    def test_handles_noisy_image(self):
        noisy = _make_noisy_pil()
        out = preprocess_for_ocr(noisy)
        assert isinstance(out, Image.Image)


class TestLoadImageFromBytes:
    def test_roundtrip_png(self):
        pil = _make_gray_pil(100, 100)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        loaded = load_image_from_bytes(buf.getvalue())
        assert loaded.size == pil.size

    def test_roundtrip_jpeg(self):
        pil = _make_gray_pil(80, 120)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        loaded = load_image_from_bytes(buf.getvalue())
        assert loaded.size == pil.size
