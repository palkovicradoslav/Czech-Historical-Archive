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

sys.path.insert(1, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_PATH = os.path.normpath(
    os.path.join(BASE_DIR, '..', 'pages', 'structured_records')
)
INDEX_DIR = os.path.normpath(
    os.path.join(BASE_DIR, 'vital_records_index')
)
LIMIT = 10
IMAGES_DIR = os.path.normpath(
    os.path.join(BASE_DIR, '..', 'pages', 'images')
)
CROPPED_IMAGES_DIR = os.path.normpath(
    os.path.join(BASE_DIR, 'static', 'images')
)


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


def get_region_coordinates(pagexml_path, region_id):
    """Extract coordinates for a specific region from a PageXML file"""
    try:
        if not os.path.exists(pagexml_path):
            logging.error(f"PageXML file not found: {pagexml_path}")
            return None

        tree = ET.parse(pagexml_path)
        root = tree.getroot()

        ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}

        text_region = root.find(f".//ns:TextRegion[@id='{region_id}']", ns)
        if text_region is None:
            return None

        coords_elem = text_region.find(".//ns:Coords", ns)
        if coords_elem is not None:
            return coords_elem.get('points')

        return None
    except Exception as e:
        logging.error(f"Error getting region coordinates: {str(e)}")
        return None


def compute_bounding_box(all_coords, padding=0):
    points = []
    for pair in all_coords.split():
        x_str, y_str = pair.split(",")
        points.append((int(float(x_str)), int(float(y_str))))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x = max(min(xs) - padding, 0)
    min_y = max(min(ys) - padding, 0)
    max_x = max(xs) + padding
    max_y = max(ys) + padding
    return min_x, min_y, max_x, max_y


def create_cropped_image(record):
    """
    Crops and saves an image for a given record.
    Returns the filename of the cropped image or None on failure.
    """
    try:
        image_name = unicodedata.normalize(
            'NFD', f"{record['file'].split('/')[-1].split('_')[0]}/{record['file'].split('/')[-1].split('_', 1)[-1].rstrip('_ocr.xml').rstrip('_pp.xml')}.jpg")

        cropped_image_filename = f"{record['id']}.jpg"
        image_path = os.path.join(CROPPED_IMAGES_DIR, cropped_image_filename)

        if os.path.exists(image_path):
            logging.info(f"Cropped image already exists: {image_path}")
            return cropped_image_filename

        region_coords = get_region_coordinates(
            record['file'], record['region_id'])
        if not region_coords:
            return None

        x1, y1, x2, y2 = compute_bounding_box(region_coords, padding=0)

        full_image_path = os.path.join(IMAGES_DIR, image_name)
        if not os.path.exists(full_image_path):
            logging.error(f"Could not load image: {full_image_path}")
            return None

        img = cv2.imread(full_image_path)
        if img is None:
            logging.error(f"cv2.imread failed for image: {full_image_path}")
            return None

        # Only create the image if it doesn't exist
        if not os.path.exists(image_path):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cropped_region = img_rgb[y1:y2, x1:x2]
            cv2.imwrite(image_path, cv2.cvtColor(
                cropped_region, cv2.COLOR_RGB2BGR))

        return cropped_image_filename
    except Exception as e:
        logging.error(
            f"Error creating cropped image for record {record.get('id')}: {e}")
        return None


class VitalRecordsIndexer:
    """
    Indexes HTR'd vital records (marriages, births, deaths) from JSON files for efficient searching
    """

    def __init__(self, index_dir=INDEX_DIR):
        self.index_dir = index_dir
        # Create a custom analyzer that tokenizes, lowercases, and removes diacritics
        custom_analyzer = DiacriticRemovingAnalyzer() | LowercaseFilter()

        self.schema = Schema(
            id=ID(stored=True, unique=True, analyzer=custom_analyzer),
            record_type=TEXT(stored=True),  # "marriage", "birth", or "death"

            # Marriage record fields
            wedding_date=TEXT(stored=True, analyzer=custom_analyzer),
            wedding_place=TEXT(stored=True, analyzer=custom_analyzer),
            bride_name=TEXT(stored=True, analyzer=custom_analyzer),
            bride_surname=TEXT(stored=True, analyzer=custom_analyzer),
            bride_mother=TEXT(stored=True, analyzer=custom_analyzer),
            bride_father=TEXT(stored=True, analyzer=custom_analyzer),
            bride_birthplace=TEXT(stored=True, analyzer=custom_analyzer),
            bride_birthday=TEXT(stored=True, analyzer=custom_analyzer),
            groom_name=TEXT(stored=True, analyzer=custom_analyzer),
            groom_surname=TEXT(stored=True, analyzer=custom_analyzer),
            groom_mother=TEXT(stored=True, analyzer=custom_analyzer),
            groom_father=TEXT(stored=True, analyzer=custom_analyzer),
            groom_birthplace=TEXT(stored=True, analyzer=custom_analyzer),
            groom_birthday=TEXT(stored=True, analyzer=custom_analyzer),
            groom_id=TEXT(stored=True, analyzer=custom_analyzer),
            bride_id=TEXT(stored=True, analyzer=custom_analyzer),

            # Birth record fields
            name=TEXT(stored=True, analyzer=custom_analyzer),
            surname=TEXT(stored=True, analyzer=custom_analyzer),
            birthplace=TEXT(stored=True, analyzer=custom_analyzer),
            birthdate=TEXT(stored=True, analyzer=custom_analyzer),
            mother=TEXT(stored=True, analyzer=custom_analyzer),
            father=TEXT(stored=True, analyzer=custom_analyzer),
            mother_place_of_living=TEXT(stored=True, analyzer=custom_analyzer),
            father_place_of_living=TEXT(stored=True, analyzer=custom_analyzer),

            # Birth and death records share this field
            person_id=TEXT(stored=True, analyzer=custom_analyzer),

            # Death record fields
            date_of_death=TEXT(stored=True, analyzer=custom_analyzer),
            place_of_living=TEXT(stored=True, analyzer=custom_analyzer),
            reason_of_death=TEXT(stored=True, analyzer=custom_analyzer),
            additional_info=TEXT(stored=True, analyzer=custom_analyzer),

            # Common fields
            region_id=TEXT(stored=True),
            file=TEXT(stored=True),
            source=STORED,
            cropped_image_url=STORED,
            full_text=TEXT(stored=False, analyzer=custom_analyzer)
        )

        os.makedirs(index_dir, exist_ok=True)

        self.ix = create_in(index_dir, self.schema)

        self.field_weights = {
            'name': 5.0,
            'surname': 5.0,
            'bride_name': 5.0,
            'bride_surname': 5.0,
            'groom_name': 5.0,
            'groom_surname': 5.0,

            'birthdate': 1.5,
            'bride_birthday': 1.5,
            'groom_birthday': 1.5,
            'bride_birthplace': 1.5,
            'groom_birthplace': 1.5,
            'birthplace': 1.5,
            'place_of_living': 1.5,

            'bride_mother': 1.0,
            'bride_father': 1.0,
            'groom_mother': 1.0,
            'groom_father': 1.0,
            'mother': 1.0,
            'father': 1.0,

            'wedding_place': 0.5,
            'wedding_date': 0.5,
            'date_of_death': 0.5,
            'mother_place_of_living': 0.2,
            'father_place_of_living': 0.2,

            'reason_of_death': 0.2,
            'additional_info': 0.001,  # very small effect
        }

        self.fields = list(self.field_weights.keys())

    def _extract_fields_by_record_type(self, record_data, record_type):
        """Extract the appropriate fields based on record type"""
        empty_fields = {}
        for field_name in self.schema.names():
            if field_name not in ['id', 'record_type', 'source', 'full_text']:
                empty_fields[field_name] = ''

        if record_type == 'marriage':
            if record_data.get('bride_name', '') in (None, '', 'null') and \
                    record_data.get('bride_surname', '') in (None, '', 'null') and \
                    record_data.get('groom_name', '') in (None, '', 'null') and \
                    record_data.get('groom_surname', '') in (None, '', 'null'):
                return {}
            return {
                **empty_fields,
                'wedding_date': record_data.get('wedding_date', ''),
                'wedding_place': record_data.get('wedding_place', ''),
                'bride_name': record_data.get('bride_name', ''),
                'bride_surname': record_data.get('bride_surname', ''),
                'bride_mother': record_data.get('bride_mother', ''),
                'bride_father': record_data.get('bride_father', ''),
                'bride_birthplace': record_data.get('bride_birthplace', ''),
                'bride_birthday': record_data.get('bride_birthday', ''),
                'groom_name': record_data.get('groom_name', ''),
                'groom_surname': record_data.get('groom_surname', ''),
                'groom_mother': record_data.get('groom_mother', ''),
                'groom_father': record_data.get('groom_father', ''),
                'groom_birthplace': record_data.get('groom_birthplace', ''),
                'groom_birthday': record_data.get('groom_birthday', ''),
                'groom_id': record_data.get('groom_id', ''),
                'bride_id': record_data.get('bride_id', ''),
            }
        elif record_type == 'birth':
            if record_data.get('name', '') in (None, '', 'null') and \
                    record_data.get('surname', '') in (None, '', 'null'):
                return {}
            return {
                **empty_fields,
                'name': record_data.get('name', ''),
                'surname': record_data.get('surname', ''),
                'birthplace': record_data.get('birthplace', ''),
                'birthdate': record_data.get('birthdate', ''),
                'mother': record_data.get('mother', ''),
                'father': record_data.get('father', ''),
                'mother_place_of_living': record_data.get('mother_place_of_living', ''),
                'father_place_of_living': record_data.get('father_place_of_living', ''),
                'person_id': record_data.get('person_id', ''),
            }
        elif record_type == 'death':
            if record_data.get('name', '') in (None, '', 'null'):
                return {}
            return {
                **empty_fields,
                'name': record_data.get('name', ''),
                'date_of_death': record_data.get('date_of_death', ''),
                'place_of_living': record_data.get('place_of_living', ''),
                'mother': record_data.get('mother', ''),
                'father': record_data.get('father', ''),
                'reason_of_death': record_data.get('reason_of_death', ''),
                'additional_info': record_data.get('additional_info', ''),
                'person_id': record_data.get('person_id', ''),
            }
        else:
            # Return all fields as empty for unknown record types
            return empty_fields

    def build_index(self, input_dir_or_file, batch_size=100, max_workers=None):
        """Build index using parallel processing for better performance"""
        if max_workers is None:
            max_workers = min(4, multiprocessing.cpu_count())

        total_records = 0

        if os.path.isdir(input_dir_or_file):
            json_files = [os.path.join(input_dir_or_file, f)
                          for f in os.listdir(input_dir_or_file)
                          if f.endswith('.json')]

            # Process files in batches
            file_batches = [json_files[i:i + batch_size]
                            for i in range(0, len(json_files), batch_size)]

            logging.info(
                f"Processing {len(json_files)} files in {len(file_batches)} batches")

            # Use ThreadPoolExecutor for I/O bound JSON processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                batch_results = list(tqdm(
                    executor.map(self.process_json_file, file_batches),
                    total=len(file_batches),
                    desc="Processing file batches"
                ))

            # Batch write to index for better performance
            # Increase memory limit
            writer = self.ix.writer(limitmb=512, procs=max_workers)

            try:
                for batch_documents in tqdm(batch_results, desc="Writing to index"):
                    for doc in batch_documents:
                        writer.add_document(**doc)
                        total_records += 1

                    # Commit periodically to avoid memory issues
                    if total_records % 1000 == 0:
                        writer.commit()
                        writer = self.ix.writer(limitmb=512, procs=max_workers)

                writer.commit()

            except Exception as e:
                writer.cancel()
                raise e

        else:
            # Single file processing
            documents = self.process_json_file([input_dir_or_file])

            writer = self.ix.writer()
            for doc in documents:
                writer.add_document(**doc)
                total_records += 1
            writer.commit()

        logging.info(f"Indexing complete. Total records: {total_records}")
        return total_records

    def process_json_file(self, json_files_batch):
        """Process a single JSON file containing vital records"""
        all_documents = []
        for json_file_path in json_files_batch:
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle the nested JSON format where keys are record IDs
                if isinstance(data, dict):
                    count = 0

                    for record_id, record_data in data.items():
                        try:
                            record_type = record_data.get('record_type', '')

                            document_fields = self._extract_fields_by_record_type(
                                record_data, record_type)

                            if document_fields == {}:
                                logging.warning(
                                    f"Skipping empty/incomplete record {record_id} in {json_file_path}")
                                continue

                            # Add file and coordinates data
                            document_fields['file'] = record_data.get(
                                'file', '')
                            document_fields['region_id'] = record_data.get(
                                'region_id', '')
                            document_fields['source'] = json_file_path
                            document_fields['id'] = record_id
                            document_fields['record_type'] = record_type
                            cropped_filename = create_cropped_image(
                                document_fields)

                            document_fields['cropped_image_url'] = cropped_filename if cropped_filename else ''

                            # Create full text field for searching across all fields
                            document_fields['full_text'] = " ".join([
                                str(value) for key, value in document_fields.items()
                                if key not in ['id', 'source', 'file', 'region_id', 'record_type', 'person_id']
                            ])

                            all_documents.append(document_fields)
                            count += 1
                        except Exception as e:
                            logging.error(
                                f"Error processing record {record_id} in {json_file_path}: {e}")
                            continue

                else:
                    logging.warning(
                        f"Expected a dictionary in {json_file_path}, but got {type(data).__name__}")
                    return 0

            except Exception as e:
                logging.error(f"Error processing file {json_file_path}: {e}")
                return 0

        return all_documents

    @lru_cache(maxsize=1000)
    def search(self, query_string, field=None, record_type=None, limit=10):
        """
        Search the index for the given query string
        """
        query_string = unicodedata.normalize('NFD', query_string)

        # Only add fuzzy matching for non-date fields
        is_date_field = field and any(
            date_term in field for date_term in ['date', 'day', 'time'])
        if not is_date_field:
            query_string = f"{query_string}~2"

        with self.ix.searcher() as searcher:
            if field and field in self.schema.names():
                parser = QueryParser(field, schema=self.ix.schema)
            else:
                parser = MultifieldParser(
                    list(self.fields), schema=self.ix.schema, fieldboosts=self.field_weights)

            parser.add_plugin(FuzzyTermPlugin())

            query = parser.parse(query_string)

            if record_type:
                filter_query = QueryParser(
                    "record_type", schema=self.ix.schema).parse(record_type)
                results = searcher.search(
                    query, filter=filter_query, limit=limit)
            else:
                results = searcher.search(query, limit=limit)

            return [dict(r) for r in results]

    def search_by_name(self, name, record_type=None, limit=10):
        """
        Specialized search for names across different record types
        """
        name = f"{unicodedata.normalize('NFD', name)}~2"

        with self.ix.searcher() as searcher:
            # Define name fields based on record types we're interested in
            name_fields = ["bride_name", "bride_surname", "groom_name",
                           "groom_surname", "name", "surname", "mother", "father"]
            parser = MultifieldParser(
                name_fields, schema=self.ix.schema,  fieldboosts=self.field_weights)
            parser.add_plugin(FuzzyTermPlugin())
            query = parser.parse(name)

            # Add filter for record type if specified
            if record_type:
                filter_query = QueryParser(
                    "record_type", schema=self.ix.schema).parse(record_type)
                results = searcher.search(
                    query, filter=filter_query, limit=limit)
            else:
                results = searcher.search(query, limit=limit)

            return [dict(r) for r in results]

    def search_by_place(self, place, record_type=None, limit=10):
        """
        Search for a place across all place-related fields
        """
        place = f"{unicodedata.normalize('NFD', place)}~2"

        with self.ix.searcher() as searcher:
            place_fields = [
                "wedding_place", "bride_birthplace", "groom_birthplace",
                "birthplace", "mother_place_of_living", "father_place_of_living",
                "place_of_living"
            ]
            parser = MultifieldParser(
                place_fields, schema=self.ix.schema,  fieldboosts=self.field_weights)
            parser.add_plugin(FuzzyTermPlugin())
            query = parser.parse(place)

            # Add filter for record type if specified
            if record_type:
                filter_query = QueryParser(
                    "record_type", schema=self.ix.schema).parse(record_type)
                results = searcher.search(
                    query, filter=filter_query, limit=limit)
            else:
                results = searcher.search(query, limit=limit)

            return [dict(r) for r in results]

    def search_by_date(self, date_string, record_type=None, limit=10):
        """
        Search for records by date across all date fields
        """
        date_string = unicodedata.normalize('NFD', date_string)

        with self.ix.searcher() as searcher:
            date_fields = [
                "wedding_date", "bride_birthday", "groom_birthday",
                "birthdate", "date_of_death"
            ]
            parser = MultifieldParser(
                date_fields, schema=self.ix.schema,  fieldboosts=self.field_weights)
            query = parser.parse(date_string)

            # Add filter for record type if specified
            if record_type:
                filter_query = QueryParser(
                    "record_type", schema=self.ix.schema).parse(record_type)
                results = searcher.search(
                    query, filter=filter_query, limit=limit)
            else:
                results = searcher.search(query, limit=limit)

            return [dict(r) for r in results]
