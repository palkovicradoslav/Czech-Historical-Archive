import os
import sys
import xml.etree.ElementTree as ET
import unicodedata
import requests
import json
import logging
import argparse
from enum import Enum
import time

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_SRC = os.path.abspath(os.path.join(THIS_DIR, '..'))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, REPO_SRC)

from utils import correct_llm_output, setup_logger, get_api_keys  # NOQA

ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}

MAX_RETRIES = 5

role = "You are an expert assistant specialized in extracting structured data from OCR‑ed historical Czech records from 19th century using the context of all provided text."

url = "https://openrouter.ai/api/v1/chat/completions"

API_KEYS = get_api_keys()


class RecordType(Enum):
    birth = 'birth'
    death = 'death'
    marriage = 'marriage'

    def __str__(self):
        return self.value

# rest of file remains the same, defaults updated where appropriate


def parse_args():
    """Parse command-line arguments for structured record extraction pipeline."""
    parser = argparse.ArgumentParser(
        description='A Python pipeline for extracting structured information from OCR\'d text via a LLM.'
    )

    parser.add_argument(
        '--input-folder',
        type=str,
        default=os.path.join(REPO_ROOT, 'data', 'recognition_results'),
        help='Folder with the recognized text in the PageXML format.'
    )

    parser.add_argument(
        '--output-folder',
        type=str,
        default=os.path.join(REPO_ROOT, 'data', 'structured_records'),
        help='Output folder for the structured records.'
    )

    parser.add_argument(
        '--model',
        type=str,
        # suitable alternative "openai/gpt-oss-120b:free"
        default="qwen/qwen-2.5-72b-instruct:free",
        help='OpenRouter model for extracting information out of records.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    setup_logger()

    args = parse_args()

    for file in os.listdir(args.input_folder):
        if file.endswith("_ocr.xml"):
            logging.info(f"Processing file: {file}")
            basename = os.path.basename(file).rstrip('_ocr.xml')
            post_processed_file = os.path.join(
                args.input_folder, basename + "_pp.xml")
            output_path = os.path.join(
                args.output_folder, basename + "_parsed_records.json")

            OR_API_KEY = API_KEYS.get('OPENROUTER_API_KEY', None)
            if OR_API_KEY is None or OR_API_KEY.startswith('your_openrouter_key_here'):
                logging.error(
                    'No OpenRouter API key provided. Please set it in .env file')
                sys.exit(1)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OR_API_KEY}"
            }

            if os.path.exists(output_path):
                logging.info(
                    f"Extracted information {output_path} already exists. Skipping processing.")
                continue

            # rather use post processed file if available
            if os.path.exists(post_processed_file):
                file = post_processed_file
                logging.info(
                    f"Found post-processed file for original file {file}")
            else:
                file = os.path.join(args.input_folder, file)

            # call process_page_xml (implementation kept from original)
            process_page_xml(
                file, basename, basename.split('_')[0], output_path, headers, model=args.model)
            logging.info(
                f"Processing of {file} completed successfully!\n" + "-"*40)
            time.sleep(1)  # to avoid hitting the API too fast
