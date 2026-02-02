#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "$SCRIPT_DIR"

echo "Starting the worker..."
docker-compose up -d ocr-worker

echo "Step 1: Running text recognition pipeline..."
docker-compose exec ocr-worker python recognition/pipeline.py --post-processing

echo "Step 2: Running structured information extraction..."
docker-compose exec ocr-worker python extraction/structured_records_extraction.py