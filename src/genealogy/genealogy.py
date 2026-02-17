import json
from datetime import datetime
from jarowinkler import jarowinkler_similarity
import re
from unidecode import unidecode
import unicodedata
import os
from collections import defaultdict
import math

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

RECORDS_DIR = os.path.normpath(os.path.join(
    REPO_ROOT, 'data', 'structured_records'))

GEN_RECORDS_DIR = os.path.normpath(os.path.join(
    REPO_ROOT, 'data', 'genealogy_structured_records'))

# Minimum plausible generation gap (parent → child)
MIN_PARENT_MARRIAGE_AGE = 14
# Defensive approach to avoid improbable parent ages
MAX_PARENT_MARRIAGE_AGE = 50
# Fuzzy-match threshold for considering two names identical
NAME_SIMILARITY_THRESHOLD = 0.9
# Probabilistic match threshold
MATCH_PROBABILITY_THRESHOLD = 0.85

# rest of original implementation preserved
