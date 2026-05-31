"""Celery configuration.

Redis plays two roles here:
  - broker:  the queue FastAPI drops jobs into
  - backend: where finished results are stored for the client to poll
"""
import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ocr_extractor",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker"],
)

celery_app.conf.update(
    task_serializer="pickle",       # we pass raw file bytes to the task
    accept_content=["pickle", "json"],
    result_serializer="json",       # results are plain JSON-friendly dicts
    result_expires=3600,            # drop results after an hour
    task_track_started=True,
)
