"""Image preprocessing pipeline that runs before OCR.

Raw images from scanners, phones, or scanned PDFs are often skewed,
noisy, or low-contrast. Tesseract works best on clean, straight,
high-contrast binary images. This module converts whatever we receive
into that ideal form.

Pipeline (applied in order):
  1. Grayscale
  2. Scale up  -- Tesseract accuracy drops sharply below ~150 dpi
  3. Denoise   -- removes salt-and-pepper noise from cheap scanners
  4. Deskew    -- corrects tilt introduced by placing a doc unevenly
  5. Adaptive threshold -- handles uneven lighting / shadows
"""
import io
import logging

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Tesseract is happiest around 300 dpi. If the image is tiny, scaling
# it up makes characters large enough to recognize reliably.
_MIN_HEIGHT_PX = 1000
_SCALE_FACTOR = 2.0

# Deskew only fires when the detected angle exceeds this. Sub-half-degree
# tilts are not worth the interpolation blur they introduce.
_DESKEW_ANGLE_THRESHOLD = 0.5


def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    """Main entry point. Accepts a PIL image, returns a cleaned PIL image."""
    arr = _to_gray(pil_image)
    arr = _scale_if_small(arr)
    arr = _denoise(arr)
    arr = _deskew(arr)
    arr = _adaptive_threshold(arr)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Internal steps
# ---------------------------------------------------------------------------

def _to_gray(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _scale_if_small(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < _MIN_HEIGHT_PX:
        scale = _SCALE_FACTOR
        new_w, new_h = int(w * scale), int(h * scale)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        log.debug("Scaled image from %dx%d to %dx%d", w, h, new_w, new_h)
    return arr


def _denoise(arr: np.ndarray) -> np.ndarray:
    # fastNlMeansDenoising is effective for document noise without
    # blurring the thin strokes of text. h=10 is a mild setting.
    return cv2.fastNlMeansDenoising(arr, h=10, templateWindowSize=7, searchWindowSize=21)


def _deskew(arr: np.ndarray) -> np.ndarray:
    """Detect the dominant text angle and rotate to correct it.

    Strategy: invert + dilate to merge nearby characters into blobs,
    then fit a minimum-area rectangle to those blobs. The rectangle's
    angle is the skew angle.
    """
    inverted = cv2.bitwise_not(arr)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    dilated = cv2.dilate(inverted, kernel, iterations=1)

    coords = np.column_stack(np.where(dilated > 0))
    if len(coords) < 50:
        return arr

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0). Map to a human-readable skew.
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) < _DESKEW_ANGLE_THRESHOLD:
        return arr

    log.debug("Deskewing by %.2f degrees", angle)
    h, w = arr.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        arr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _adaptive_threshold(arr: np.ndarray) -> np.ndarray:
    """Convert to binary. Adaptive thresholding handles documents that
    are darker in one corner (common with phone photos)."""
    return cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=11,
    )


# ---------------------------------------------------------------------------
# Utility: load from raw bytes without knowing the format up front
# ---------------------------------------------------------------------------

def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))
