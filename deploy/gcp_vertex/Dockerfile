FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libxml2-dev libxslt1-dev libjpeg-dev zlib1g-dev libssl-dev \
    git wget pkg-config \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install numpy==1.26.4

# Copy only what the recognition server needs
COPY src/recognition/ /app/src/recognition/
COPY src/utils.py     /app/src/utils.py
COPY deploy/gcp_vertex/server.py /app/server.py

ENV PYTHONPATH=/app/src
ENV DATA_DIR=/app/data

EXPOSE 8080

ENV REGION_MODEL_PATH="/app/src/recognition/models/model_regions.mlmodel"
ENV LINE_MODEL_PATH="/app/src/recognition/models/model_lines.mlmodel"
ENV RECOG_MODEL_PATH="/app/src/recognition/models/model_recognition.mlmodel"
ENV DEVICE=cpu

# Vertex AI health checks probe AIP_HTTP_PORT (default 8080)
ENV AIP_HTTP_PORT=8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]