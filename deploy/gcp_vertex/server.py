from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import Response, JSONResponse
import os
import sys
import shutil
import tempfile
import logging
from typing import Optional
from pydantic import BaseModel
from google.cloud import storage
import pathlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from recognition import pipeline  # NOQA

app = FastAPI(title="Kraken Recognition Server")

LOGGER = logging.getLogger("uvicorn.error")

# Global model handles
MODELS = {
    'region_model': None,
    'line_model': None,
    'recog_model': None,
    'processor': None,
}


def load_models_from_env():
    device = os.environ.get('DEVICE', 'cpu')
    region_path = os.environ.get('REGION_MODEL_PATH')
    line_path = os.environ.get('LINE_MODEL_PATH')
    recog_path = os.environ.get('RECOG_MODEL_PATH')

    # If paths are provided but files are missing treat them as None
    if region_path and not os.path.exists(region_path):
        logging.warning(
            f"Region model path provided but not found: {region_path}. Skipping region model.")
        region_path = None
    if line_path and not os.path.exists(line_path):
        logging.warning(
            f"Line model path provided but not found: {line_path}. Skipping line model.")
        line_path = None
    if recog_path and not os.path.exists(recog_path):
        logging.warning(
            f"Recognition model path provided but not found: {recog_path}. Skipping recognition model.")
        recog_path = None

    region_model, line_model, recog_model, processor = pipeline.load_models(
        region_path, line_path, recog_path, device=device
    )
    MODELS['region_model'] = region_model
    MODELS['line_model'] = line_model
    MODELS['recog_model'] = recog_model
    MODELS['processor'] = processor


@app.on_event('startup')
def startup_event():
    logging.basicConfig(level=logging.INFO)
    LOGGER.info('Starting server and loading models (CPU)...')
    load_models_from_env()
    LOGGER.info('Models loaded')


@app.get('/')
async def root():
    return JSONResponse(status_code=200, content={'status': 'ok'})


@app.get('/health')
async def health():
    return JSONResponse(status_code=200, content={'status': 'ok'})


@app.get('/v1/endpoints/{endpoint_id}/deployedModels/{deployed_model_id}')
async def vertex_health(endpoint_id: str, deployed_model_id: str):
    return JSONResponse(status_code=200, content={'status': 'ok'})


@app.post('/predict')
async def predict(record_type: str = Form('birth'), post_processing: bool = Form(False), file: UploadFile = File(...)):
    if record_type not in {'birth', 'death', 'marriage'}:
        raise HTTPException(
            status_code=400, detail='record_type must be one of birth, death, marriage')

    # Ensure models loaded
    if MODELS['recog_model'] is None and MODELS['processor'] is None:
        raise HTTPException(status_code=500, detail='Models not loaded')

    tmpdir = tempfile.mkdtemp()
    try:
        input_path = os.path.join(tmpdir, file.filename)
        with open(input_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        output_dir = os.path.join(tmpdir, 'out')
        os.makedirs(output_dir, exist_ok=True)

        # call pipeline.process_file which handles segmentation + recognition
        pipeline.process_file(
            input_path,
            record_type,
            output_dir,
            MODELS['region_model'],
            MODELS['line_model'],
            MODELS['recog_model'],
            MODELS['processor'],
            post_processing=post_processing
        )

        basename = os.path.splitext(os.path.basename(input_path))[0]
        ocr_xml = os.path.join(output_dir, f"{record_type}_{basename}_ocr.xml")
        if not os.path.isfile(ocr_xml):
            raise HTTPException(
                status_code=500, detail='OCR output not generated')

        with open(ocr_xml, 'rb') as f:
            data = f.read()

        return Response(content=data, media_type='application/xml')
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            logging.error("Error cleaning up temp directory")


class GcsRequest(BaseModel):
    gcs_uri: Optional[str] = None
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    record_type: str = 'birth'
    post_processing: bool = False
    upload_output: bool = True


def download_blob_to_file(bucket_name: str, blob_name: str, dest_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(dest_path)


def upload_file_to_blob(bucket_name: str, blob_name: str, src_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(src_path)


@app.post('/predict_gcs')
async def predict_gcs(req: GcsRequest):
    # ensure models loaded
    if MODELS['recog_model'] is None and MODELS['processor'] is None:
        raise HTTPException(status_code=500, detail='Models not loaded')

    if not req.gcs_uri and not (req.bucket and req.prefix is not None):
        raise HTTPException(
            status_code=400, detail='Provide either gcs_uri or bucket+prefix')

    client = storage.Client()

    if req.gcs_uri:
        # parse gs://bucket/path
        if not req.gcs_uri.startswith('gs://'):
            raise HTTPException(
                status_code=400, detail='gcs_uri must start with gs://')
        parts = req.gcs_uri[5:].split('/', 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ''

        suffix = pathlib.Path(blob_path).suffix
        if suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            raise HTTPException(
                status_code=400, detail='gcs_uri must point to an image')

        tmpdir = tempfile.mkdtemp()
        try:
            local_image = os.path.join(tmpdir, os.path.basename(blob_path))
            download_blob_to_file(bucket_name, blob_path, local_image)

            output_dir = os.path.join(tmpdir, 'out')
            os.makedirs(output_dir, exist_ok=True)

            pipeline.process_file(
                local_image,
                req.record_type,
                output_dir,
                MODELS['region_model'],
                MODELS['line_model'],
                MODELS['recog_model'],
                MODELS['processor'],
                post_processing=req.post_processing
            )

            basename = os.path.splitext(os.path.basename(local_image))[0]
            ocr_xml = os.path.join(
                output_dir, f"{req.record_type}_{basename}_ocr.xml")
            if not os.path.isfile(ocr_xml):
                raise HTTPException(
                    status_code=500, detail='OCR output not generated')

            if req.upload_output:
                out_blob = f"{pathlib.Path(blob_path).parent}/{basename}_ocr.xml"
                # place outputs in a sibling folder 'recognition_results' if parent is non-empty
                parent = pathlib.Path(blob_path).parent
                if str(parent) == '.' or str(parent) == '':
                    out_blob = f"recognition_results/{basename}_ocr.xml"
                else:
                    out_blob = f"{parent}_recognition_results/{basename}_ocr.xml"

                upload_file_to_blob(bucket_name, out_blob, ocr_xml)
                gcs_out_uri = f"gs://{bucket_name}/{out_blob}"
                return JSONResponse(status_code=200, content={"output_uri": gcs_out_uri})
            else:
                with open(ocr_xml, 'rb') as f:
                    data = f.read()
                return Response(content=data, media_type='application/xml')
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                logging.error("Error cleaning up temp directory")

    # bucket + prefix mode: process all images under prefix and upload outputs
    uploaded = []
    bucket_name = req.bucket
    prefix = req.prefix
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        name = blob.name
        if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        tmpdir = tempfile.mkdtemp()
        try:
            local_image = os.path.join(tmpdir, os.path.basename(name))
            blob.download_to_filename(local_image)
            output_dir = os.path.join(tmpdir, 'out')
            os.makedirs(output_dir, exist_ok=True)

            pipeline.process_file(
                local_image,
                req.record_type,
                output_dir,
                MODELS['region_model'],
                MODELS['line_model'],
                MODELS['recog_model'],
                MODELS['processor'],
                post_processing=req.post_processing
            )

            basename = os.path.splitext(os.path.basename(local_image))[0]
            ocr_xml = os.path.join(
                output_dir, f"{req.record_type}_{basename}_ocr.xml")
            if os.path.isfile(ocr_xml) and req.upload_output:
                parent = pathlib.Path(name).parent
                if str(parent) == '.' or str(parent) == '':
                    out_blob = f"recognition_results/{basename}_ocr.xml"
                else:
                    out_blob = f"{parent}_recognition_results/{basename}_ocr.xml"
                upload_file_to_blob(bucket_name, out_blob, ocr_xml)
                uploaded.append(f"gs://{bucket_name}/{out_blob}")
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                logging.error("Error cleaning up temp directory")

    return JSONResponse(status_code=200, content={"uploaded": uploaded})
