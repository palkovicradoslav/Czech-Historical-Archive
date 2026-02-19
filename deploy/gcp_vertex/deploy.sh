#!/usr/bin/env bash

# Deploy script for GCP Vertex AI (traditional model + endpoint)

PROJECT=${PROJECT:-}
REGION=${REGION:-europe-west1}
IMAGE_NAME=kraken-recognition
TAG=${TAG:-v1}
MODEL_DISPLAY_NAME=kraken-recognition-model
ENDPOINT_DISPLAY_NAME=kraken-recognition-endpoint

if [ -z "$PROJECT" ]; then
  echo "Please set PROJECT environment variable (GCP project id)"
  exit 1
fi

ARTIFACT_REPO=ocr-repo
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${ARTIFACT_REPO}/${IMAGE_NAME}:${TAG}"

echo "Create Artifact Registry"
gcloud artifacts repositories describe "${ARTIFACT_REPO}" --location="${REGION}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${ARTIFACT_REPO}" --repository-format=docker --location="${REGION}" --description="Docker repo for kraken recognition" \
  || true

gcloud auth configure-docker "${REGION}-docker.pkg.dev" -q

echo "Building container"
docker build -f gcp_deployment/Dockerfile -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"

echo "Uploading model to Vertex AI"
MODEL_ID=$(gcloud ai models upload --region=${REGION} --container-image-uri=${IMAGE_URI} --display-name=${MODEL_DISPLAY_NAME} --container-predict-route="/predict_gcs" --format="value(name)")

echo "Creating endpoint"
ENDPOINT_ID=$(gcloud ai endpoints create --region=${REGION} --display-name=${ENDPOINT_DISPLAY_NAME} --format="value(name)")

echo "Deploying model to endpoint"
gcloud ai endpoints deploy-model "${ENDPOINT_ID}" \
  --region=${REGION} \
  --model="${MODEL_ID}" \
  --display-name="${MODEL_DISPLAY_NAME}-deployment" \
  --machine-type=n1-standard-4 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --traffic-split=0=100

echo "gcloud ai endpoints predict --endpoint=ENDPOINT_ID --region=${REGION} --json-request=request.json"
