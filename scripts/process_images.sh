#!/usr/bin/env bash
echo "Starting the worker container (detached)..."
docker compose up -d ocr-worker

echo "Running text recognition pipeline inside worker..."
docker compose exec ocr-worker python -m recognition.pipeline --post-processing

echo "Running structured records extraction inside worker..."
docker compose exec ocr-worker python -m extraction.structured_records_extraction

echo "Processing finished."
