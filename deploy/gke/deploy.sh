#!/bin/bash
set -euo pipefail

# config
PROJECT=${PROJECT:?"PROJECT environment variable must be set"}
REGION=europe-west1
ZONE=europe-west1-b
CLUSTER_NAME=historical-ocr-cluster
ARTIFACT_REPO=ocr-repo
TAG=${TAG:-$(date)}
WEBAPP_IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${ARTIFACT_REPO}/historical-ocr-app:${TAG}"

# Navigate to repository root
cd "$(dirname "$0")/../.."

echo "Checking cluster..."
if ! gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
  echo "Creating cluster..."
  gcloud container clusters create "${CLUSTER_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --num-nodes=1 \
    --machine-type=e2-standard-2 \
    --disk-size=30 \
    --enable-ip-alias
fi

echo "Getting credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT"

# Ensure GKE node SA can pull from Artifact Registry
echo "Setting IAM permissions..."
PROJECT_NUM=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
NODE_SA=${NODE_SA:-${PROJECT_NUM}-compute@developer.gserviceaccount.com}
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:$NODE_SA" \
  --role="roles/artifactregistry.reader" >/dev/null

echo "==> Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories create "${ARTIFACT_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Docker repository for Historical Archive OCR app" 2>/dev/null || true

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "Setting up Artifact Registry..."
gcloud artifacts repositories create "$ARTIFACT_REPO" \
  --repository-format=docker \
  --location="$REGION" 2>/dev/null || true
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "Building and pushing image..."
docker build -t "$WEBAPP_IMAGE" -f deploy/gke/Dockerfile.app .
docker push "$WEBAPP_IMAGE"

echo "Deploying..."
cd deploy/gke
kustomize build . | sed -e "s|PROJECT_ID|$PROJECT|g" -e "s|IMAGE_TAG|$TAG|g" | kubectl apply -f -

echo "Rollout"
kubectl -n historical-ocr rollout status deployment/ocr-app --timeout=120s
kubectl -n historical-ocr get pods
