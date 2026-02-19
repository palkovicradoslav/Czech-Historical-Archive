FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev libxslt1-dev libjpeg-dev zlib1g-dev libssl-dev git wget pkg-config \
 && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data

# Expose port for uvicorn
EXPOSE 8080

ENV REGION_MODEL_PATH="src/recognition/models/model_regions.mlmodel"
ENV LINE_MODEL_PATH="src/recognition/models/model_lines.mlmodel"
ENV RECOG_MODEL_PATH="src/recognition/models/model_recognition.mlmodel"
ENV DEVICE=cpu

# Run the FastAPI server
CMD ["uvicorn", "deploy/gcp_vertex/server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]