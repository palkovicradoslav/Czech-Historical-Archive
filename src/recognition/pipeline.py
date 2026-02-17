import logging
import argparse
import time
import unicodedata
import pylev
from collections import Counter
from google.genai import types
from google import genai
from pydantic import BaseModel, Field
import ast
import json
import requests
from kraken.lib.xml import XMLPage
from kraken.lib.util import is_bitonal
from kraken.lib.progress import KrakenProgressBar
from kraken.lib.dataset import PolygonGTDataset, collate_sequences, ImageInputTransforms
from kraken.lib.segmentation import extract_polygons
from kraken.lib import vgsl, models
from kraken.containers import Segmentation, BaselineLine
from kraken import blla, serialization
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader
from shapely.geometry import Polygon, box
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np
import xml.etree.ElementTree as ET
from functools import partial
import os
import sys
import shutil

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_SRC = os.path.abspath(os.path.join(THIS_DIR, '..'))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, REPO_SRC)

DICTS_DIR = os.path.normpath(os.path.join(THIS_DIR, 'dictionaries'))

from utils import correct_llm_output, get_api_keys  # NOQA

ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defaults and parse_args updated to use data/ and src/ paths


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='A Python pipeline for recognizing text from full-page images of historical registries.'
    )

    parser.add_argument(
        '--verbose',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Enable verbose output'
    )

    parser.add_argument(
        '--post-processing',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Option to include post-processing - if True at least one API key has to be provided'
    )

    parser.add_argument(
        '--input-folder',
        type=str,
        default=os.path.join(REPO_ROOT, 'data', 'images'),
        help='Path to the input folder.'
    )

    parser.add_argument(
        '--output-folder',
        type=str,
        default=os.path.join(REPO_ROOT, 'data', 'recognition_results'),
        help='Path to the input file.'
    )

    parser.add_argument(
        '--text-line-segmentation-model',
        type=str,
        default=os.path.join(REPO_SRC, 'recognition',
                             'models', 'model_lines.mlmodel'),
        help='Path to the text line segmentation model.'
    )

    parser.add_argument(
        '--text-region-segmentation-model',
        type=str,
        default=os.path.join(REPO_SRC, 'recognition',
                             'models', 'model_regions.mlmodel'),
        help='Path to the text region segmentation model.'
    )

    parser.add_argument(
        '--text-recognition-model',
        type=str,
        default=os.path.join(REPO_SRC, 'recognition',
                             'models', 'model_recognition.mlmodel'),
        help='Path to the text recognition model. Can be either kraken .mlmodel or TrOCR model folder'
    )

    if len(sys.argv) <= 1:
        parser.print_help()

    return parser.parse_args()

# rest of the file remains unchanged
