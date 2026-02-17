import json
import os
import sys
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser, MultifieldParser, FuzzyTermPlugin
from tqdm import tqdm
from whoosh.analysis import Analyzer, Tokenizer, RegexTokenizer, LowercaseFilter
import logging
from unidecode import unidecode
import unicodedata
import xml.etree.ElementTree as ET
import cv2
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import lru_cache

# Ensure src/ is on path for imports
REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_SRC)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
INPUT_PATH = os.path.normpath(os.path.join(DATA_DIR, 'structured_records'))
INDEX_DIR = os.path.normpath(os.path.join(BASE_DIR, 'vital_records_index'))
LIMIT = 10
IMAGES_DIR = os.path.normpath(os.path.join(DATA_DIR, 'images'))
CROPPED_IMAGES_DIR = os.path.normpath(
    os.path.join(BASE_DIR, 'static', 'images'))


class DiacriticRemovingTokenizer(Tokenizer):
    def __init__(self, base_tokenizer=None):
        self.base_tokenizer = RegexTokenizer()

    def __call__(self, text, **kwargs):
        # Remove diacritics before tokenization
        cleaned_text = unidecode(text)
        for token in self.base_tokenizer(cleaned_text, **kwargs):
            yield token


class DiacriticRemovingAnalyzer(Analyzer):
    def __init__(self, base_tokenizer=None):
        self.tokenizer = DiacriticRemovingTokenizer(base_tokenizer)

    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)

# rest of file kept identical to original implementation
