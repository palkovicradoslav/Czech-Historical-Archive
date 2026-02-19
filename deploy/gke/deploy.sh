PROJECT=${PROJECT:-}
REGION=${REGION:-europe-west1}
ARTIFACT_REPO=ocr-app-repo
IMAGE_NAME=kraken-webapp
TAG=${TAG:-latest}
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${ARTIFACT_REPO}/${IMAGE_NAME}:${TAG}"

gcloud artifacts repositories create ${ARTIFACT_REPO} --repository-format=docker --location=${REGION} --description="Docker repository for OCR app"

gcloud auth configure-docker ${REGION}-docker.pkg.dev

docker build -t ${IMAGE_URI} -f dockerfile.cloudrun .

docker push ${IMAGE_URI}

gcloud run deploy ${IMAGE_NAME} \
  --image=${IMAGE_URI} \
  --platform=managed \
  --port=5000 \
  --memory=4Gi \
  --allow-unauthenticated \
  --region=${REGION} \
  --max 3