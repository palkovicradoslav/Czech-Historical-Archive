#!/usr/bin/env bash
set -euo pipefail

# Run from the project root: bash deploy/gcp_vertex/deploy.sh

PROJECT=${GCP_PROJECT:?GCP_PROJECT not set}
REGION=${GCP_REGION:-europe-west1}
TAG=${TAG:-v1}
MACHINE_TYPE=${MACHINE_TYPE:-n1-standard-4}
GCS_BUCKET=${GCP_BUCKET:?GCP_BUCKET not set}
ARTIFACT_REPO=ocr-repo
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${ARTIFACT_REPO}/kraken-recognition:${TAG}"
PREDICTION_SA=${VERTEX_SERVICE_ACCOUNT:?VERTEX_SERVICE_ACCOUNT not set}

# service account
gcloud iam service-accounts describe "${PREDICTION_SA}" --project="${PROJECT}" >/dev/null 2>&1 \
  || gcloud iam service-accounts create kraken-prediction-sa --display-name="Kraken Prediction SA" --project="${PROJECT}"

gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="serviceAccount:${PREDICTION_SA}" --role="roles/storage.objectAdmin"

PROJECT_NUMBER=$(gcloud projects describe "${PROJECT}" --format="value(projectNumber)")
gcloud iam service-accounts add-iam-policy-binding "${PREDICTION_SA}" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator" --project="${PROJECT}"

# build and push image
gcloud artifacts repositories describe "${ARTIFACT_REPO}" --location="${REGION}" --project="${PROJECT}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${ARTIFACT_REPO}" --repository-format=docker --location="${REGION}" --project="${PROJECT}"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" -q
docker build -f deploy/gcp_vertex/Dockerfile -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"

# register model
MODEL_ID=$(gcloud ai models upload \
  --region="${REGION}" --project="${PROJECT}" \
  --display-name=kraken-recognition-model \
  --container-image-uri="${IMAGE_URI}" \
  --container-predict-route="/vertex_predict" \
  --container-health-route="/health" \
  --container-ports=8080 \
  --format="value(name)")

# create endpoint and deploy
ENDPOINT_ID=$(gcloud ai endpoints create \
  --region="${REGION}" --project="${PROJECT}" \
  --display-name=kraken-recognition-endpoint \
  --format="value(name)")

gcloud ai endpoints deploy-model "${ENDPOINT_ID}" \
  --region="${REGION}" --project="${PROJECT}" \
  --model="${MODEL_ID}" \
  --display-name=kraken-recognition-deployment \
  --machine-type="${MACHINE_TYPE}" \
  --min-replica-count=1 --max-replica-count=1 \
  --traffic-split=0=100 \
  --service-account="${PREDICTION_SA}"

echo "Done. Endpoint: ${ENDPOINT_ID}"
