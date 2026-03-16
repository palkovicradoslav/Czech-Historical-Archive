from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from google.cloud import storage
import os
import shutil
import tempfile
import logging
import pathlib
from recognition import pipeline

log = logging.getLogger("uvicorn.error")

MODELS = {'region_model': None, 'line_model': None,
          'recog_model': None, 'processor': None}


def load_models():
    device = os.environ.get('DEVICE', 'cpu')
    paths = {
        'region': os.environ.get('REGION_MODEL_PATH'),
        'line':   os.environ.get('LINE_MODEL_PATH'),
        'recog':  os.environ.get('RECOG_MODEL_PATH'),
    }
    missing = [k for k, v in paths.items() if not v or not os.path.exists(v)]
    if missing:
        raise RuntimeError(f"Model files missing or not set: {missing}")
    rm, lm, rec, proc = pipeline.load_models(
        paths['region'], paths['line'], paths['recog'], device=device)
    MODELS.update(region_model=rm, line_model=lm,
                  recog_model=rec, processor=proc)


@asynccontextmanager
async def lifespan(app):
    logging.basicConfig(level=logging.INFO)
    log.info('Loading models...')
    load_models()
    log.info('Models loaded')
    yield


app = FastAPI(title="Kraken Recognition Server", lifespan=lifespan)


def gcs_output_path(blob_path, basename):
    parent = str(pathlib.Path(blob_path).parent)
    if parent in ('.', ''):
        return f"recognition_results/{basename}_ocr.xml"
    return f"{parent}_recognition_results/{basename}_ocr.xml"


def run_pipeline(image_path, record_type, output_dir, post_processing=False):
    pipeline.process_file(
        image_path, record_type, output_dir,
        MODELS['region_model'], MODELS['line_model'],
        MODELS['recog_model'], MODELS['processor'],
        post_processing=post_processing,
    )
    basename = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, f"{record_type}_{basename}_ocr.xml"), basename


@app.get('/health')
async def health():
    return JSONResponse({'status': 'ok'})


class VertexInstance(BaseModel):
    gcs_uri: str
    record_type: str = 'birth'
    post_processing: bool = False
    upload_output: bool = True


class VertexRequest(BaseModel):
    instances: list[VertexInstance]


@app.post('/vertex_predict')
async def vertex_predict(req: VertexRequest):
    predictions = []
    for inst in req.instances:
        if not inst.gcs_uri.startswith('gs://'):
            predictions.append({'error': 'gcs_uri must start with gs://'})
            continue
        bucket_name, blob_path = inst.gcs_uri[5:].split('/', 1)
        if pathlib.Path(blob_path).suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            predictions.append({'error': 'gcs_uri must point to an image'})
            continue
        tmpdir = tempfile.mkdtemp()
        try:
            local_image = os.path.join(tmpdir, os.path.basename(blob_path))
            storage.Client().bucket(bucket_name).blob(
                blob_path).download_to_filename(local_image)
            output_dir = os.path.join(tmpdir, 'out')
            os.makedirs(output_dir)
            ocr_xml, basename = run_pipeline(
                local_image, inst.record_type, output_dir, inst.post_processing)
            if not os.path.isfile(ocr_xml):
                predictions.append({'error': 'OCR output not generated'})
                continue
            if inst.upload_output:
                out_blob = gcs_output_path(blob_path, basename)
                storage.Client().bucket(bucket_name).blob(
                    out_blob).upload_from_filename(ocr_xml)
                predictions.append(
                    {'output_uri': f'gs://{bucket_name}/{out_blob}'})
            else:
                predictions.append({'xml': open(ocr_xml).read()})
        except Exception as e:
            log.exception("Error processing instance")
            predictions.append({'error': str(e)})
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return JSONResponse({'predictions': predictions})
