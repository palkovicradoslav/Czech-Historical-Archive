FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lighter than the full requirements
COPY deploy/gke/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/utils.py          /app/
COPY src/genealogy/        /app/genealogy/
COPY src/app/              /app/app/
COPY src/app/records_indexer.py /app/

WORKDIR /app

ENV DATA_DIR=/app/data

EXPOSE 5000

# Production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "600", "app.app:app"]
