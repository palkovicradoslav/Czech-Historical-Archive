import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import aiplatform, storage

load_dotenv()

PROJECT = os.environ["GCP_PROJECT"]
REGION = os.environ.get("GCP_REGION", "europe-west1")
BUCKET = os.environ["GCP_BUCKET"]
ENDPOINT_ID = os.environ["VERTEX_ENDPOINT_ID"]
MODEL_ID = os.environ["VERTEX_MODEL_ID"]
SERVICE_ACCOUNT = os.environ["VERTEX_SERVICE_ACCOUNT"]

ENDPOINT_NAME = f"projects/{PROJECT}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
MODEL_NAME = f"projects/{PROJECT}/locations/{REGION}/models/{MODEL_ID}"


def predict_online(gcs_uri, record_type="birth", post_processing=False, upload_output=True):
    aiplatform.init(project=PROJECT, location=REGION)
    endpoint = aiplatform.Endpoint(ENDPOINT_NAME)
    response = endpoint.predict(instances=[{
        "gcs_uri": gcs_uri,
        "record_type": record_type,
        "post_processing": post_processing,
        "upload_output": upload_output,
    }], timeout=600)
    return response.predictions[0]


def predict_batch(local_inputs_jsonl, output_gcs_dir, wait=True):
    aiplatform.init(project=PROJECT, location=REGION)

    input_gcs_uri = f"gs://{BUCKET}/batch_inputs/{Path(local_inputs_jsonl).name}"
    parts = input_gcs_uri.removeprefix("gs://").split("/", 1)
    storage.Client().bucket(parts[0]).blob(
        parts[1]).upload_from_filename(local_inputs_jsonl)

    job = aiplatform.BatchPredictionJob.create(
        job_display_name=f"kraken-batch-{int(time.time())}",
        model_name=MODEL_NAME,
        gcs_source=input_gcs_uri,
        gcs_destination_prefix=output_gcs_dir,
        instances_format="jsonl",
        predictions_format="jsonl",
        machine_type="n1-standard-4",
        starting_replica_count=1,
        max_replica_count=1,
        service_account=SERVICE_ACCOUNT,
    )

    if wait:
        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        while job.state.name not in terminal:
            print(f"{job.state.name}")
            time.sleep(30)
            job._sync_gca_resource()
        print(f"  {job.state.name}")

    return job


def get_batch_results(output_gcs_dir):
    """Read prediction results from GCS and return list of output URIs."""
    parts = output_gcs_dir.removeprefix("gs://").split("/", 1)
    bucket_name, prefix = parts[0], parts[1].rstrip("/") + "/"

    results = []
    for blob in storage.Client().list_blobs(bucket_name, prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        for line in blob.download_as_text().splitlines():
            if line.strip():
                results.append(json.loads(line))
    return results


if __name__ == "__main__":
    # Single image
    result = predict_online(
        gcs_uri=f"gs://{BUCKET}/inputs/your-image.jpg",
        record_type="marriage",
    )
    print(result)

    # Batch prediction
    job = predict_batch(
        local_inputs_jsonl="deploy/gcp_vertex/inputs.jsonl",
        output_gcs_dir=f"gs://{BUCKET}/batch_outputs/",
        wait=True,
    )

    for r in get_batch_results(f"gs://{BUCKET}/batch_outputs/"):
        print(r)
