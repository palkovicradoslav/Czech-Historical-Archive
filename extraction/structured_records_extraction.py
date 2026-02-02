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
sys.path.insert(1, os.path.join(THIS_DIR, '..'))

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


def normalize_json(data):
    """Normalize data by applying NFD normalization."""
    if isinstance(data, str):
        return unicodedata.normalize('NFD', data)
    elif isinstance(data, dict):
        return {key: normalize_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [normalize_json(item) for item in data]
    return data


def extract_wedding_record_details(ocr_text, model, headers):
    """Extract structured marriage details from HTR text using an LLM prompt."""
    prompt = f"""
You are given OCR text from a historical Czech marriage record from 19th century. Extract and return the following details in a JSON object:
- wedding_date: the date of the wedding
- wedding_place: the place of the wedding
- groom_name: the given name of groom
- groom_surname: the family name of groom
- groom_father: groom's father's full name
- groom_father_place: groom's father's place of living
- groom_mother: groom's mother's full name
- groom_mother_place: groom's mother's place of living
- groom_birthplace: place of birth of groom
- groom_birthday: date of birth of groom
- bride_name: the given name of bride
- bride_surname: the family name of bride
- bride_father: bride's father's full name
- bride_father_place: bride's father's place of living
- bride_mother: bride's mother's full name
- bride_mother_place: bride's mother's place of living
- bride_birthplace: place of birth of bride
- bride_birthday: date of birth of bride

Your task is to find the information using the context from all the provided text.
If neccessary modify the word forms of names of people and places to be in basic word form.
Return your output strictly as JSON with keys "wedding_date", "wedding_place", "groom_name", "groom_surname", "groom_father",
"groom_father_place", "groom_mother", "groom_mother_place", "groom_birthplace", "groom_birthday",
"bride_name", "bride_surname", "bride_father", "bride_father_place", "bride_mother", "bride_mother_place", "bride_birthplace", and "bride_birthday".
If a value is not available, use null.

---
### Example

OCR text:

7. Dne 14ho Cervna, 1870. II,, 6. ho III ,, 12 ho června, 1870. Antonín Holub, rojín na dovolené od c. k. pešího pluku H ství Mělníckého;
a jeho manželky + Kateřiny, ro- zené Malákovy ze Střem- /: Narodil se dne 10. dubna, 1846.: Ev č. 28. pozustalý syn po + Jiřím Holubovi;
domkáři ze Lhotky okresní hejtm, č. d. 17. okresního hejtman- Mhělnícké. Hrazko č. d. 10. c. k. OhláIdne 29. května šene bgle.
1 1/2 1. Anna Balounova, Ev. 21 manželská dcera Ji-H. řího Balouna, dom-C. káře z Hrazka č. d. 10. okresního hejt- manství Mělnícké- ho;
a jeho manžel- ky Anny rozené Koutkovy ze Šedlce č.d. 33. 1: 612. c. Zjř dnvn Oddával: EdMolnár Hrazka, č.d. 12.
/: Patřící listiny jsou v archivu ćrkve pod Daclav Rařslit. C. fol. LX + III čís. 7. :/ domkář z Hrazka č.d. 4. Žeoter neplnolelé norǐsty
kořatkaf tom to u přítomnosti výse psaných svěd.- domkár z farár. Kü svolil, svědcí podpis jeho;
/: Narodila se dne 9. hvěna, 1849:/ Drm Sialuřnc otec

Output:
```json
{{
  "wedding_date": "1870-06-14",
  "wedding_place": "Hrazko",
  "groom_name": "Antonín",
  "groom_surname": "Holub",
  "groom_father": "Jiří Holub",
  "groom_father_place": "Lhotka",
  "groom_mother": "Kateřina Maláková",
  "groom_mother_place": "Střemy",
  "groom_birthplace": "Lhotka",
  "groom_birthday": "1846-04-10",
  "bride_name": "Anna",
  "bride_surname": "Balounová",
  "bride_father": "Jiří Baloun",
  "bride_father_place": "Hrazko",
  "bride_mother": "Anna Koutková",
  "bride_mother_place": "Sedlce",
  "bride_birthplace": "Hrazko",
  "bride_birthday": "1849-05-09"
}}

---

Now process the real OCR text.
Do not include any additional text.

OCR text:
{json.dumps(ocr_text, ensure_ascii=False, indent=2)}
"""
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "marriage_record_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "wedding_date": {
                        "type": "string",
                        "description": "Date the wedding took place"
                    },
                    "wedding_place": {
                        "type": "string",
                        "description": "Place where the wedding happened"
                    },
                    "bride_name": {
                        "type": "string",
                        "description": "Given name of the bride"
                    },
                    "bride_surname": {
                        "type": "string",
                        "description": "Surname of the bride"
                    },
                    "bride_father": {
                        "type": "string",
                        "description": "Bride's father's name"
                    },
                    "bride_father_place": {
                        "type": "string",
                        "description": "Bride's father's place of living"
                    },
                    "bride_mother": {
                        "type": "string",
                        "description": "Bride's mother's name"
                    },
                    "bride_mother_place": {
                        "type": "string",
                        "description": "Bride's mother's place of living"
                    },
                    "bride_birthplace": {
                        "type": "string",
                        "description": "Place of birth of bride"
                    },
                    "bride_birthday": {
                        "type": "string",
                        "description": "Birthday of bride"
                    },
                    "groom_name": {
                        "type": "string",
                        "description": "Given name of the groom"
                    },
                    "groom_surname": {
                        "type": "string",
                        "description": "Surname of the groom"
                    },
                    "groom_father": {
                        "type": "string",
                        "description": "Groom's father's name"
                    },
                    "groom_father_place": {
                        "type": "string",
                        "description": "Groom's father's place of living"
                    },
                    "groom_mother": {
                        "type": "string",
                        "description": "Groom's mother's name"
                    },
                    "groom_mother_place": {
                        "type": "string",
                        "description": "Groom's mother's place of living"
                    },
                    "groom_birthplace": {
                        "type": "string",
                        "description": "Place of birth of groom"
                    },
                    "groom_birthday": {
                        "type": "string",
                        "description": "Birthday of groom"
                    }
                },
                "required": [
                    "wedding_date",
                    "wedding_place",
                    "bride_name",
                    "bride_surname",
                    "bride_father",
                    "bride_father_place",
                    "bride_mother",
                    "bride_mother_place",
                    "bride_birthplace",
                    "bride_birthday",
                    "groom_name",
                    "groom_surname",
                    "groom_father",
                    "groom_father_place",
                    "groom_mother",
                    "groom_mother_place",
                    "groom_birthplace",
                    "groom_birthday"
                ],
                "additionalProperties": False
            }
        }
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": role
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "response_format": response_format
    }

    # Send the POST request to the OpenRouter API.
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return correct_llm_output(response)


def extract_birth_record_details(ocr_text, model, headers):
    """Extract structured birth details from HTR text using an LLM prompt."""
    prompt = f"""
You are given OCR text from a historical Czech birth record from the 19th century. Extract and return the following details in a JSON object:
- name: first name of the individual
- surname: family name of the individual
- birthplace: place where the individual was born
- birthdate: date of birth of the individual
- father: full name of the individual's father
- mother: full name of the individual's mother
- father_place_of_living: individual's father's place of living
- mother_place_of_living: individual's mother's place of living

Your task is to find the information using the context from all the provided text.
If necessary modify the word forms of names and places to be in their basic form.
Return your output strictly as JSON with keys "name", "surname", "birthplace", "birthdate", "father", "mother", "father_place_of_living", and "mother_place_of_living".
If a value is not available, use null.

---
### Example

OCR text:

32 EdMolnár farář. 0/2 havd. od- půt. dne dne Střem. c. k.. okresní 13. 15. hejtman. Října, Mělnícké. 1876. 37. V chrámu Far. Páne evang
Nebuddřec- ském. Jan 1. 11 man žel- ké Josef Holub domkář a Knížecí Ev hájný ve Střemách č. d. 37. c.k. okres, C hejtm. Mělnícké-
ho, syn po + Jiřímy Holubovi, domká- ri ze Lhotky č. d. 17; a jeho v manželky + Ka- teřiny rodem MMalákovy ze Střem Alžběta,
rozená dcera po + Antoní- nu Počepkovi domkáři ze Lhob-. ký č.d. 2, c.k. okřes. hejtm. Mělníckého; a jeho manželky V Marie,
rodem Mokré z Chodče 5̌. d. 1.- svíbodná dcera n Bazivkova manželka Josefa Růžička rolníka z Penžchova č.0. Marie po + Josef Herman- Bǒhmnová vi,
rolníku ze Střem č.d. 23. Kak Ev Ana Heřmananá H. z Nebudžele zkoušená

Output:
```json
{{
  "name": "Jan",
  "surname": "Holub",
  "birthplace": "Střemy",
  "birthdate": "1876-10-13",
  "father": "Josef Holub",
  "father_place_of_living": "Střemy",
  "mother": "Alžběta Počepková",
  "mother_place_of_living": "Střemy"
}}

---

Now process the real OCR text.
Do not include any additional text.

OCR text:
{ocr_text}
"""
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "birth_record_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "First name of the individual"
                    },
                    "surname": {
                        "type": "string",
                        "description": "Family name of the individual"
                    },
                    "birthplace": {
                        "type": "string",
                        "description": "Place where the individual was born"
                    },
                    "birthdate": {
                        "type": "string",
                        "description": "Date of birth of the individual"
                    },
                    "father": {
                        "type": "string",
                        "description": "Full name of the individual's father"
                    },
                    "father_place_of_living": {
                        "type": "string",
                        "description": "Individual's father's place of living"
                    },
                    "mother": {
                        "type": "string",
                        "description": "Full name of the individual's mother"
                    },
                    "mother_place_of_living": {
                        "type": "string",
                        "description": "Individual's mother's place of living"
                    }
                },
                "required": [
                    "name",
                    "surname",
                    "birthplace",
                    "birthdate",
                    "father",
                    "father_place_of_living",
                    "mother",
                    "mother_place_of_living"
                ],
                "additionalProperties": False
            }
        }
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": role
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "response_format": response_format
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return correct_llm_output(response)


def extract_death_record_details(ocr_text, model, headers):
    """Extract structured death details from HTR text using an LLM prompt."""
    prompt = f"""
You are given OCR text from a historical Czech death record from the 19th century. Extract and return the following details in a JSON object:
- name: full name of the deceased
- date_of_death: the date of death
- place_of_living: the place where the deceased was living
- father: full name of the deceased's father
- father_place_of_living: the deceased's father's place of living
- mother: full name of the deceased's mother
- mother_place_of_living: the deceased's mother's place of living
- reason_of_death: reason or cause of death
- additional_info: additional important information

Your task is to find the information using the context from all the provided text.
If necessary, modify the word forms of names and places to be in basic word form. Do not include any additional text.
Return your output strictly as JSON with keys "name", "date_of_death", "place_of_living", "father",  "father_place_of_living",
"mother", "mother_place_of_living", "reason_of_death", and "additional_info".
If a value is not available, use null.

---
### Example

OCR text:

29  O 8nie hodině ráno 4 ho dne 56. Velnó- ho Oujezd 1861. Zare, Antonín Luňak, manželský , synácék Jana Luňáka,
domkáře z Velk. Oujezda 26 č.d. 56. c. k. okresa Mělnického. a jeho mánželk Anny rodemé Rytiřovy z Hestíny čd. 54.
,, Asotník dd Repín z ho záři, 1801. EdMolnár místní evg. farář. 1. Póchován na evang. hrbitově v Nebudželi 2.
Narodil se due 8. srpna, 1861.

Output:
```json
{{
  "name": "Antonín Luňák",
  "date_of_death": "1861-09-04",
  "place_of_living": "Velký Oujezd",
  "father": "Jan Luňák",
  "father_place_of_living": "Velký Oujezd",
  "mother": "Anna Rytířová",
  "mother_place_of_living": "Hostína",
  "reason_of_death": "Psotník",
  "additional_info": "Buried in Nebudžely; born on 1861-08-08"
}}

---

Now process the real OCR text.
Do not include any additional text.

OCR text:
{ocr_text}
"""
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "death_record_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Full name of the deceased"
                    },
                    "date_of_death": {
                        "type": "string",
                        "description": "Date of death"
                    },
                    "place_of_living": {
                        "type": "string",
                        "description": "Place where the deceased was living at the time of the record"
                    },
                    "father": {
                        "type": "string",
                        "description": "Full name of the deceased's father"
                    },
                    "father_place_of_living": {
                        "type": "string",
                        "description": "The deceased's father's place of living"
                    },
                    "mother": {
                        "type": "string",
                        "description": "Full name of the deceased's mother"
                    },
                    "mother_place_of_living": {
                        "type": "string",
                        "description": "The deceased's mother's place of living"
                    },
                    "reason_of_death": {
                        "type": "string",
                        "description": "Reason or cause of death"
                    },
                    "additional_info": {
                        "type": "string",
                        "description": "Additional important information"
                    }
                },
                "required": [
                    "name",
                    "date_of_death",
                    "place_of_living",
                    "father",
                    "father_place_of_living",
                    "mother",
                    "mother_place_of_living",
                    "reason_of_death",
                    "additional_info"
                ],
                "additionalProperties": False
            }
        }
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": role
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "response_format": response_format
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return correct_llm_output(response)


def process_page_xml(input_file, output_file, file_type, output_path, headers, model):
    """Process a PageXML file, extract regions and call record extraction for each region."""
    tree = ET.parse(input_file)
    root = tree.getroot()

    records = []

    for region in root.findall(".//ns:TextRegion", ns):
        region_id = region.get("id", "N/A")

        region_type = region.attrib.get('type', 'Unknown')
        region_type = region.get('custom', 'structure {type:Unknown;}').split(
            'structure {type:')[1].strip(';}')
        if region_type == 'Header':
            logging.info(f"Skipping Header region {region_id}")
            continue

        logging.info(f"Processing {region_type} region with id: {region_id}")

        # extract text from the region
        lines_info = []

        for textline in region.findall(".//ns:TextLine", ns):
            unicode_elem = textline.find("ns:TextEquiv/ns:Unicode", ns)
            if unicode_elem is None:
                continue
            text_value = unicode_elem.text.strip() if unicode_elem.text else ""

            text_value = unicodedata.normalize('NFD', text_value)

            lines_info.append(text_value)

        if not lines_info:
            logging.info(
                f"Region {region_id} contains no text lines; skipping.")
            continue

        record_json = None
        for retry_attempt in range(MAX_RETRIES):
            try:
                match file_type:
                    case "marriage":
                        record_json = normalize_json(
                            extract_wedding_record_details(" ".join(lines_info), model, headers))
                    case "death":
                        record_json = normalize_json(
                            extract_death_record_details(" ".join(lines_info), model, headers))
                    case "birth":
                        record_json = normalize_json(
                            extract_birth_record_details(" ".join(lines_info), model, headers))
                if record_json is None:
                    wait_time = (retry_attempt + 1) ** 2
                    logging.warning(
                        f"Processing region {region_id} failed. Waiting {wait_time}")
                    time.sleep(wait_time)  # wait before retrying
                    logging.warning("Retrying...")
                else:
                    break
            except Exception as e:
                wait_time = (retry_attempt + 1) ** 2
                logging.warning(
                    f"Processing region {region_id} failed: {e}. Waiting {wait_time}")
                # wait before retrying
                time.sleep(wait_time)
                logging.warning("Retrying...")

        if record_json is not None:
            record_json['region_id'] = region_id
            record_json['file'] = unicodedata.normalize('NFD', input_file)
            record_json['record_type'] = unicodedata.normalize(
                'NFD', file_type)

            records.append(record_json)
        else:
            logging.error(
                f"Failed to process region {region_id} after {MAX_RETRIES} retries. Skipping.")

    # output the record to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        container = {}
        for i, rec in enumerate(records):
            container[f'{output_file}_{i}'] = rec
        json.dump(container, f, ensure_ascii=False, indent=4)


def parse_args():
    """Parse command-line arguments for structured record extraction pipeline."""
    parser = argparse.ArgumentParser(
        description='A Python pipeline for extracting structured information from OCR\'d text via a LLM.'
    )

    parser.add_argument(
        '--input-folder',
        type=str,
        default='pages/recognition_results',
        help='Folder with the recognized text in the PageXML format.'
    )

    parser.add_argument(
        '--output-folder',
        type=str,
        default='pages/structured_records',
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

            process_page_xml(
                file, basename, basename.split('_')[0], output_path, headers, model=args.model)
            logging.info(
                f"Processing of {file} completed successfully!\n" + "-"*40)
            time.sleep(1)  # to avoid hitting the API too fast
