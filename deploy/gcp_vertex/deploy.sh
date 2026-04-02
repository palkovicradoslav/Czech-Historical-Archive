#!/usr/bin/env bash
set -euo pipefail

# Run from the project root: bash deploy/gcp_vertex/deploy.sh

PROJECT=${GCP_PROJECT:?GCP_PROJECT not set}
REGION=${GCP_REGION:-europe-west1}
ARTIFACT_REPO="ocr-repo"

echo "Determining tag"
LATEST_TAG=$(gcloud artifacts docker tags list \
  "${REGION}-docker.pkg.dev/${PROJECT}/${ARTIFACT_REPO}/historical-ocr-app" \
  --limit=1 --sort-by="~UPDATE_TIME" --format="value(tag)" 2>/dev/null || echo "v0")

TAG_NUMBER=$(echo "${LATEST_TAG#v}" | grep -oE '^[0-9]+' || echo 0)
NEXT_TAG="v$((TAG_NUMBER + 1))"

echo "Tag: $NEXT_TAG"
TAG=$NEXT_TAG

MACHINE_TYPE=${MACHINE_TYPE:-n1-standard-4}
GCS_BUCKET=${GCP_BUCKET:?GCP_BUCKET not set}

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${ARTIFACT_REPO}/kraken-recognition:${TAG}"
PREDICTION_SA=${VERTEX_SERVICE_ACCOUNT:-kraken-prediction-sa@${PROJECT}.iam.gserviceaccount.com}
PREDICTION_SA_ACCOUNT_ID="${PREDICTION_SA%@*}"

# service account
gcloud iam service-accounts describe "${PREDICTION_SA}" --project="${PROJECT}" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${PREDICTION_SA_ACCOUNT_ID}" --display-name="Kraken Prediction SA" --project="${PROJECT}"

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

# register model (reuse if already exists)
MODEL_ID=$(gcloud ai models list \
  --region="${REGION}" --project="${PROJECT}" \
  --filter='displayName="kraken-recognition-model"' \
  --sort-by='~createTime' --limit=1 \
  --format="value(name)")

if [[ -z "${MODEL_ID}" ]]; then
  MODEL_ID=$(gcloud ai models upload \
    --region="${REGION}" --project="${PROJECT}" \
    --display-name=kraken-recognition-model \
    --container-image-uri="${IMAGE_URI}" \
    --container-predict-route="/vertex_predict" \
    --container-health-route="/health" \
    --container-ports=8080 \
    --format="value(name)")
fi

# create endpoint and deploy (reuse if already exists)
ENDPOINT_ID=$(gcloud ai endpoints list \
  --region="${REGION}" --project="${PROJECT}" \
  --filter='displayName="kraken-recognition-endpoint"' \
  --sort-by='~createTime' --limit=1 \
  --format="value(name)")

if [[ -z "${ENDPOINT_ID}" ]]; then
  ENDPOINT_ID=$(gcloud ai endpoints create \
    --region="${REGION}" --project="${PROJECT}" \
    --display-name=kraken-recognition-endpoint \
    --format="value(name)")
fi

gcloud ai endpoints deploy-model "${ENDPOINT_ID}" \
  --region="${REGION}" --project="${PROJECT}" \
  --model="${MODEL_ID}" \
  --display-name=kraken-recognition-deployment \
  --machine-type="${MACHINE_TYPE}" \
  --min-replica-count=1 --max-replica-count=1 \
  --traffic-split=0=100 \
  --service-account="${PREDICTION_SA}"

echo "MODEL_ID=${MODEL_ID}"
echo "ENDPOINT_ID=${ENDPOINT_ID}"
echo "Done"
