#!/bin/bash
set -euo pipefail

# Get the image tag from running deployment or use provided one
IMAGE_TAG=${1:-$(kubectl -n historical-ocr describe pod -l app=ocr-app | grep "Image:" | head -1 | awk -F: '{print $NF}' | xargs)}

PROJECT_ID=${PROJECT_ID:?"PROJECT_ID environment variable must be set"}

echo "Using image tag: $IMAGE_TAG"

# Delete any existing job first
kubectl -n historical-ocr delete job build-index-job 2>/dev/null || true

# Apply job with substitutions
sed -e "s|PROJECT_ID|$PROJECT_ID|g" -e "s|IMAGE_TAG|$IMAGE_TAG|g" job-build-index.yaml | kubectl apply -f -

echo "Job created. Monitor with:"
echo "  kubectl -n historical-ocr get jobs"
echo "  kubectl -n historical-ocr logs -f job/build-index-job"
