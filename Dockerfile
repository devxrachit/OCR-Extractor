FROM python:3.11-slim

# System dependencies:
#   tesseract-ocr  -> the OCR engine
#   poppler-utils  -> needed by pdf2image to render PDFs
#   libgl1          -> required by opencv-python-headless at import time
#   libglib2.0-0   -> same (OpenCV links against GLib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
