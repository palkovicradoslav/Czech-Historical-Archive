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
DICTS_DIR = os.path.normpath(os.path.join(THIS_DIR, 'dictionaries'))
sys.path.insert(1, os.path.join(THIS_DIR, '..'))

from utils import correct_llm_output, get_api_keys  # NOQA

ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_list = [
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "qwen/qwen3-30b-a3b:free",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-pro",
    "deepseek/deepseek-chat-v3-0324:free"

    # were good, but no longer working correctly
    # "qwen/qwen3-32b:free",
    # "meta-llama/llama-4-maverick:free",
    # "nvidia/llama-3.3-nemotron-super-49b-v1:free",
    # "gemini-1.5-flash",
]

role = "You are an expert assistant that corrects OCR typographical and spelling mistakes on historical Czech records from 19th century using the context of all provided textlines."

few_shot_examples = """
Example 1:
Input: {"id": "A", "text": "dne 7.čevon"}
Output: {"id": "A", "corrected_text": "dne 7.června"}

Example 2:
Input: {"id": "B", "text": "domkav̌e z Kvotelce Česko Brodkéhl"}
Output: {"id": "B", "corrected_text": "domkáře z Kvotelce Česko Brodského"}

Example 3:
Input: {"id": "C", "text": "Joznamka: Umotní listek"}
Output: {"id": "C", "corrected_text": "Poznamka: Umrtní lístek"}
"""

API_KEYS = get_api_keys()


def segment_page_into_lines(image_path, model, file, output_path):
    """Segment a page image into lines in a PageXML using the given model."""
    if os.path.isfile(output_path):
        logging.warning(
            f"Segmentation for {file} exists\tSkipping this file...")
        return

    im = Image.open(image_path)

    baseline_seg = blla.segment(im, model=model)
    baseline_seg.imagename = os.path.basename(image_path)

    p = serialization.serialize(
        baseline_seg, image_size=im.size, template='pagexml')

    with open(output_path, 'w') as fp:
        fp.write(p)

    # pretty print
    ET.register_namespace('', ns['ns'])
    tree = ET.parse(output_path)
    root = tree.getroot()

    page_elem = root.find(".//ns:Page", ns)
    # properly set the image filename
    if page_elem is not None:
        page_elem.set('imageFilename', os.path.basename(image_path))

    ET.indent(tree)
    tree.write(output_path, xml_declaration=True, encoding='UTF-8')

    logging.info(
        f"Segmentation of {file} completed succesfully! Results written to {output_path}")


def parse_pagexml_regions(tree):
    """Parse PageXML and return a list of regions with types."""
    regions = []
    root = tree.getroot()

    page = root.find(".//ns:Page", ns)
    page_width = float(page.attrib.get('imageWidth', 0))
    page_height = float(page.attrib.get('imageHeight', 0))

    for text_region in root.findall(".//ns:TextRegion", ns):
        coords = text_region.find("ns:Coords", ns)
        if coords is not None:
            points_str = coords.attrib['points']
            points = [tuple(map(float, point.split(',')))
                      for point in points_str.split()]
            region_type = text_region.attrib.get('type', 'Unknown')
            region_type = text_region.get('custom', 'structure {type:Unknown;}').split(
                'structure {type:')[1].strip(';}')

            polygon = Polygon(points)
            # ignore full page regions
            if not is_full_page_region(polygon, page_width, page_height):
                regions.append((text_region, polygon, region_type))

    return regions


def parse_pagexml_text_lines_with_text(file_path):
    """Extract text lines and their polygons."""
    text_lines = []
    tree = ET.parse(file_path)

    for text_line in get_text_lines(tree):
        coords = text_line.find("ns:Coords", ns)
        if coords is not None:
            points_str = coords.attrib['points']
            points = [tuple(map(float, point.split(',')))
                      for point in points_str.split()]
            polygon = Polygon(points)

            label = ""
            textequiv = text_line.find("ns:TextEquiv", ns)
            if textequiv is not None:
                unicode_elem = textequiv.find("ns:Unicode", ns)
                if unicode_elem is not None and unicode_elem.text:
                    label = unicode_elem.text.strip()

            text_lines.append((text_line, polygon, label))

    return text_lines


def assign_textlines_to_regions(textlines, regions):
    """Assign each text line to the most appropriate region by containment or distance."""
    assignments = {reg: [] for reg in regions}

    for textline_tuple in textlines:
        textline_polygon, _ = textline_tuple
        best_region_tuple = None
        min_distance = float('inf')
        textline_centroid = textline_polygon.centroid

        contained_regions = []
        for region_tuple in regions:
            region_polygon, _ = region_tuple
            # text is completely contained in the region
            if region_polygon.contains(textline_polygon):
                contained_regions.append(region_tuple)

        if contained_regions:
            best_region_tuple = min(contained_regions,
                                    key=lambda r: r[0].area)
        else:
            # find the closest region
            for region_tuple in regions:
                region_polygon, _ = region_tuple

                distance = textline_centroid.distance(region_polygon)

                if distance < min_distance:
                    min_distance = distance
                    best_region_tuple = region_tuple

        if best_region_tuple:
            assignments[best_region_tuple].append(textline_tuple)

    return assignments


def is_full_page_region(polygon, page_width, page_height, threshold=0.99):
    """Return True if a polygon covers nearly the whole page."""
    page_area = page_width * page_height
    region_area = polygon.area

    coverage = region_area / page_area if page_area > 0 else 0

    return coverage > threshold


def remove_lines_regions_files(regions_path, lines_path):
    """Remove intermediate files."""
    try:
        os.remove(regions_path)
        os.remove(lines_path)
    except:
        logging.error("File processing failed!")


def get_filenames(file, dir_path, basename):
    """Return standard derived filenames for a basename."""
    lines_path = os.path.join(dir_path, basename + "_lines_segmented.xml")
    regions_path = os.path.join(dir_path, basename + "_regions_segmented.xml")
    image_path = os.path.join(dir_path, basename + ".jpg")
    ocr_path = os.path.join(dir_path, basename + "_ocr.xml")
    lines_regions_path = os.path.join(dir_path, basename + "_segmented.xml")
    return lines_path, regions_path, image_path, ocr_path, lines_regions_path


def get_text_lines(tree):
    """Return all text lines from PageXML."""
    root = tree.getroot()

    return [text_line for text_line in root.findall(".//ns:TextLine", ns)]


def process_pagexml(regions_path, lines_path, output_path, padding=20):
    """Assign text lines to regions and write an updated PageXML with region bounding boxes."""
    ET.register_namespace('', ns['ns'])

    tree = ET.parse(regions_path)
    regions = parse_pagexml_regions(tree)
    text_lines = parse_pagexml_text_lines_with_text(lines_path)

    region_textlines = {}

    for (orig_line, textline_polygon, label) in text_lines:
        centroid = textline_polygon.centroid
        best_region = None
        min_distance = float('inf')

        # ensure best fitting region for the text line
        contained = []
        for (region_elem, region_polygon, region_type) in regions:
            if region_polygon.contains(textline_polygon):
                contained.append((region_elem, region_polygon, region_type))

        if contained:
            best_region = min(contained, key=lambda r: r[1].area)
        else:
            for (region_elem, region_polygon, region_type) in regions:
                distance = centroid.distance(region_polygon)
                if distance < min_distance:
                    min_distance = distance
                    best_region = (region_elem, region_polygon, region_type)

        if best_region:
            region_elem = best_region[0]
            if region_elem not in region_textlines:
                region_textlines[region_elem] = []
            region_textlines[region_elem].append(textline_polygon)

            new_text_line = ET.Element(
                "{%s}TextLine" % ns['ns'], attrib=orig_line.attrib)

            orig_coords = orig_line.find("ns:Coords", ns)
            if orig_coords is not None:
                new_coords = ET.Element("{%s}Coords" %
                                        ns['ns'], attrib=orig_coords.attrib)
                new_text_line.append(new_coords)

            orig_baseline = orig_line.find("ns:Baseline", ns)
            if orig_baseline is not None:
                new_baseline = ET.Element(
                    "{%s}Baseline" % ns['ns'], attrib=orig_baseline.attrib)
                new_text_line.append(new_baseline)

            new_text_equiv = ET.Element("{%s}TextEquiv" % ns['ns'])
            new_unicode = ET.Element("{%s}Unicode" % ns['ns'])
            new_unicode.text = label
            new_text_equiv.append(new_unicode)
            new_text_line.append(new_text_equiv)

            region_elem.append(new_text_line)

    # Update each region's Coords element with a new bounding box that encloses all its text lines (with padding)
    for (region_elem, region_polygon, region_type) in regions:
        if region_elem in region_textlines:
            all_points = []
            for poly in region_textlines[region_elem]:
                all_points.extend(list(poly.exterior.coords))
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]

            min_x = max(min(xs) - padding, 0)
            min_y = max(min(ys) - padding, 0)
            max_x = max(xs) + padding
            max_y = max(ys) + padding

            new_region_polygon = box(min_x, min_y, max_x, max_y)

            points_str = " ".join([f"{int(x)},{int(y)}" for x, y in list(
                new_region_polygon.exterior.coords)[:-1]])
            coords_el = region_elem.find("ns:Coords", ns)
            if coords_el is not None:
                coords_el.attrib["points"] = points_str
            else:
                new_coords = ET.Element("{%s}Coords" % ns['ns'], attrib={
                                        "points": points_str})
                region_elem.insert(0, new_coords)

    ET.indent(tree)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    remove_lines_regions_files(regions_path, lines_path)

    logging.info(
        f"Region assignment of {lines_path} completed successfully! Results written to {output_path}")


def recognize_text_into_pagexml(file_path, output_path, geometry):
    """Put recognized text into corresponding detected text lines in PageXML file."""
    ET.register_namespace('', ns['ns'])

    tree = ET.parse(file_path)

    custom_id = 0

    for text_line in get_text_lines(tree):
        coords = text_line.find("ns:Coords", ns)
        baseline = text_line.find("ns:Baseline", ns)

        if coords is not None and baseline is not None:
            points_str = coords.attrib['points']
            polygon_points = [tuple(map(int, point.split(',')))
                              for point in points_str.split()]

            points_str = baseline.attrib['points']
            baseline_points = [tuple(map(int, point.split(',')))
                               for point in points_str.split()]
            for bl, boundary, pred in geometry:
                bl, boundary = bl[0], boundary[0]

                bl_tuple = [tuple(point) for point in bl]
                polygon_tuples = [tuple(point) for point in boundary]

                if baseline_points == bl_tuple and polygon_points == polygon_tuples:
                    text_equiv = text_line.find(
                        './/ns:TextEquiv', namespaces=ns)
                    if text_equiv is None:
                        text_equiv = ET.SubElement(text_line,
                                                   '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}TextEquiv')

                    unicode_elem = text_equiv.find(
                        './/ns:Unicode', namespaces=ns)
                    if unicode_elem is None:
                        unicode_elem = ET.SubElement(text_equiv,
                                                     '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Unicode')

                    unicode_elem.text = pred
                    break  # match found

        custom_id += 1

    ET.indent(tree)
    tree.write(output_path, xml_declaration=True, encoding='UTF-8')


def get_polygon_gen(baseline_points, polygon_points, im, custom_id):
    """Returns a generator that yields cropped polygon images"""
    baseline_line = BaselineLine(
        id=custom_id,
        baseline=baseline_points,
        boundary=polygon_points
    )

    bounds = Segmentation(type='baselines', lines=[baseline_line], imagename=None,
                          text_direction='horizontal-lr', script_detection=False)

    return extract_polygons(im, bounds)


def recognize_text_trocr(model, processor, baseline_points, polygon_points, im, custom_id):
    """Recognize text in polygons using TrOCR model"""
    gen = get_polygon_gen(baseline_points, polygon_points, im, custom_id)

    prediction = ""
    for g, b in gen:

        pixel_values = processor(
            images=g, return_tensors="pt").pixel_values.to(device)

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]

        prediction += generated_text + " "
    return prediction.strip()


def recognize_text_into_pagexml_trocr(file_path, output_path, im, recognizer, processor):
    """Recognize text in PageXML polygons using a TrOCR model."""
    if os.path.isfile(output_path):
        logging.warning(f"OCR for {output_path} exists\tSkipping this file...")
        return

    logging.info(f"Recognizing text in {file_path}")

    ET.register_namespace('', ns['ns'])

    tree = ET.parse(file_path)

    custom_id = 0

    for text_line in get_text_lines(tree):
        coords = text_line.find("ns:Coords", ns)
        baseline = text_line.find("ns:Baseline", ns)

        if coords is not None and baseline is not None:
            points_str = coords.attrib['points']
            polygon_points = [tuple(map(int, point.split(',')))
                              for point in points_str.split()]

            points_str = baseline.attrib['points']
            baseline_points = [tuple(map(int, point.split(',')))
                               for point in points_str.split()]

            recognized_text = recognize_text_trocr(
                recognizer, processor, baseline_points, polygon_points, im, custom_id)

            text_equiv = text_line.find('.//ns:TextEquiv', namespaces=ns)
            if text_equiv is None:
                text_equiv = ET.SubElement(
                    text_line, '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}TextEquiv')
            unicode_elem = text_equiv.find('.//ns:Unicode', namespaces=ns)
            if unicode_elem is None:
                unicode_elem = ET.SubElement(
                    text_equiv, '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Unicode')

            unicode_elem.text = recognized_text

        custom_id += 1

    tree.write(output_path, xml_declaration=True, encoding='UTF-8')

    logging.info(
        f"Text recognition of {file_path} completed succesfully! Results written to {output_path}")


class ExtendedPolygonGTDataset(PolygonGTDataset):
    """Custom dataset class for loading PageXML files with baselines."""

    def __getitem__(self, index):
        item = self.training_set[index]
        try:
            imagename, baseline, boundary = item[0]
            if not isinstance(imagename, Image.Image):
                im = Image.open(imagename)
            else:
                im = imagename

            segmentation = Segmentation(
                type='baselines',
                imagename=imagename,
                text_direction='horizontal-lr',
                lines=[BaselineLine(
                    'id_0', baseline=baseline, boundary=boundary)],
                script_detection=True,
                regions={},
                line_orders=[]
            )
            im, _ = next(extract_polygons(
                im, segmentation, legacy=self.legacy_polygons))
            im = self.transforms(im)
            im_mode = None
            if im.shape[0] == 3:
                im_mode = b'R'
            elif im.shape[0] == 1:
                im_mode = b'L'
            if is_bitonal(im):
                im_mode = b'1'
            with self._im_mode.get_lock():
                if im_mode is not None and im_mode > self._im_mode.value:
                    self._im_mode.value = im_mode
            if self.aug:
                im = self.aug(image=im, index=index)
            return {
                'image': im,
                'target': item[1],
                'baseline': baseline,
                'boundary': boundary
            }
        except Exception:
            self.failed_samples.add(index)
            idx = np.random.randint(0, len(self.training_set))
            return self.__getitem__(idx)


def collate_sequences_with_baselines(batch):
    """Collate a batch of sequence training examples."""
    collator = collate_sequences(batch)

    sorted_batch = sorted(
        batch, key=lambda x: x['image'].shape[2], reverse=True)

    collator['baseline'] = [x['baseline'] for x in sorted_batch]
    collator['boundary'] = [x['boundary'] for x in sorted_batch]

    return collator


batch_size = 1
pad = 16
workers = 1
threads = 1


def infer(file, batch_size, model, evaluation_files, device, pad, workers,
          threads, reorder, base_dir, normalization, format_type, mode):
    """Run inference over a dataset using kraken model."""

    predicted = []
    labels = []
    geometry = []

    legacy_polygons = False

    pin_ds_mem = False
    if device != 'cpu':
        pin_ds_mem = True

    test_set = list(evaluation_files)
    test_set = [{'page': XMLPage(
        file, filetype=format_type).to_container()} for file in test_set]
    test_set[0]['page'].imagename = file

    valid_norm = False
    DatasetClass = partial(ExtendedPolygonGTDataset,
                           legacy_polygons=legacy_polygons, skip_empty_lines=False)

    if reorder and base_dir != 'auto':
        reorder = base_dir

    with threadpool_limits(limits=threads):
        batch, channels, height, width = model.nn.input
        ts = ImageInputTransforms(
            batch, height, width, channels, (pad, 0), valid_norm, False)
        ds = DatasetClass(normalization=normalization,
                          whitespace_normalization=True,
                          reorder=reorder,
                          im_transforms=ts)
        for line in test_set:
            try:
                ds.add(**line)
            except ValueError as e:
                raise Exception(f"Loading {line} failed")

        ds.no_encode()
        ds_loader = DataLoader(ds,
                               batch_size=batch_size,
                               num_workers=workers,
                               pin_memory=pin_ds_mem,
                               collate_fn=collate_sequences_with_baselines)

        with KrakenProgressBar() as progress:
            batches = len(ds_loader)
            pred_task = progress.add_task(
                'Evaluating', total=batches, visible=True)

            for batch in ds_loader:
                im = batch['image']
                text = batch['target']
                lens = batch['seq_lens']
                baseline = batch['baseline']
                boundary = batch['boundary']

                if mode == 'eval' and text == [""]:  # empty gt
                    continue

                pred = model.predict_string(im, lens)
                for x, y in zip(pred, text):
                    predicted.append(x)
                    labels.append(y)
                    geometry.append((baseline, boundary, x))
                progress.update(pred_task, advance=1)

    return predicted, labels, geometry


def infer_page(model, file, input_file, output_file, device='cuda:0'):
    """Run text recognition on a single page and insert recognized text into PageXML."""
    if os.path.isfile(output_file):
        logging.warning(f"OCR for {input_file} exists\tSkipping this file...")
        return

    logging.info(f"Recognizing text in {input_file}")
    _, _, geometry = infer(file, batch_size, model, [input_file], device, pad, workers, threads,
                           reorder=False, base_dir=True, normalization='NFD', format_type='page', mode='infer')

    recognize_text_into_pagexml(input_file, output_file, geometry)

    logging.info(
        f"Text recognition of {input_file} completed succesfully! Results written to {output_file}")


def get_prompt(lines_info):
    """Build the LLM prompt."""
    prompt = f"""
You are provided with a JSON object containing historical Czech OCR textlines.
Each textline has an "id" and a "text" field.
Your task is to correct any spelling mistakes in each textline using the context from all provided textlines.
Please correct OCR errors in the text while preserving the original 'id' exactly as provided.
Keep corrections minimal, correct only clear insertions, substitutions, and deletions errors.
Do not complete unfinished words. If you are unsure about the correction, leave the original.
Especially leave the original if the original is probably a proper noun, starting with capital letter.
Do not modify interpunction in the original.
Below are a couple of examples of how to correct the textlines:

{few_shot_examples}

Now, apply similar corrections to the following input and return a JSON object exactly following this schema:
[
    {{
        "id": string, "corrected_text": string
    }}
]
Do not include any additional text.
Input:
{json.dumps(lines_info, ensure_ascii=False, indent=2)}
    """

    return prompt


def call_llm(lines_info, OR_API_key, model="qwen/qwen2.5-vl-72b-instruct:free"):
    """Call an OpenRouter LLM to correct HTR textlines and return structured JSON."""
    logging.info(f"Model: {model}")
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OR_API_key}"
    }

    prompt = get_prompt(lines_info)

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "ocr_correction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "textlines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Unique identifier for the textline"
                                },
                                "corrected_text": {
                                    "type": "string",
                                    "description": "The corrected version of the textline"
                                }
                            },
                            "required": ["id", "corrected_text"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["textlines"],
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

    response = requests.post(API_URL, headers=headers,
                             data=json.dumps(payload))

    return correct_llm_output(response)


class OCRCorrection(BaseModel):
    # Pydantic model for structured output
    id: str = Field(..., description="Unique identifier for the textline")
    corrected_text: str = Field(...,
                                description="The corrected version of the textline")


def call_gemini_models(lines_info, gemini_API_key, model="gemini-2.0-flash"):
    """Call Google's Gemini API to correct textlines."""
    logging.info(f"Model: {model}")

    client = genai.Client(api_key=gemini_API_key)

    prompt = get_prompt(lines_info)

    config = types.GenerateContentConfig(system_instruction=role)
    config.response_mime_type = 'application/json'
    config.response_schema = list[OCRCorrection]
    config.temperature = 1.5

    client = genai.Client(api_key=gemini_API_key)

    response = client.models.generate_content(
        model=model,
        config=config,
        contents=[prompt]
    )

    try:
        corrected_json = json.loads(response.text)
        return corrected_json
    except Exception:
        logging.warning("Failed to parse JSON output. Trying ast")
        try:
            return ast.literal_eval(response.text)
        except Exception:
            logging.error(f"Loading failed completely:\n{response}")


def post_process_page_xml(input_file, output_file):
    """Run post-processing corrections on a PageXML."""
    if os.path.isfile(output_file):
        logging.warning(
            f"Post-processing correction for {input_file} exists\tSkipping this file...")
        return

    logging.info(f"Processing {input_file}")
    tree = ET.parse(input_file)
    root = tree.getroot()

    i = 0
    for region in root.findall(".//ns:TextRegion", ns):
        region_id = region.get("id", "N/A")

        region_type = region.attrib.get('type', 'Unknown')
        region_type = region.get('custom', 'structure {type:Unknown;}').split(
            'structure {type:')[1].strip(';}')
        if region_type == 'Header':
            logging.info(f"Skipping Header region {region_id}")
            continue
        i += 1

        logging.info(f"Processing {region_type} region with id: {region_id}")

        lines_info = []
        text_line_dict = {}

        for textline in region.findall(".//ns:TextLine", ns):
            textline_id = textline.get('id')
            unicode_elem = textline.find("ns:TextEquiv/ns:Unicode", ns)
            if unicode_elem is None:
                continue
            text_value = unicode_elem.text.strip() if unicode_elem.text else ""

            text_value = unicodedata.normalize('NFD', text_value)

            lines_info.append({"id": textline_id, "text": text_value})
            text_line_dict[textline_id] = unicode_elem

        if not lines_info:
            logging.info(
                f"Region {region_id} contains no text lines; skipping.")
            continue

        corrected_items = self_consistency_correction(lines_info)

        for item in corrected_items:
            tid = item.get("id")
            corrected_text = item.get("corrected_text")
            corrected_text = unicodedata.normalize(
                'NFD', corrected_text if corrected_text is not None else "")
            if tid in text_line_dict and corrected_text is not None and text_line_dict[tid].text is not None:
                original_text = unicodedata.normalize(
                    'NFD', text_line_dict[tid].text)
                if corrected_text != original_text:
                    logging.debug(
                        f"Original:{original_text}\nCorrection:{corrected_text}")
                text_line_dict[tid].text = corrected_text

            elif tid in text_line_dict and corrected_text is None and text_line_dict[tid].text is not None:
                logging.info(
                    f"No corrected text available for {text_line_dict[tid].text}")

        time.sleep(1)

    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    logging.info(f"Updated PageXML saved to: {output_file}\n")


def read_freq_dict(dictionary_path, freq_dict):
    """Read a frequency dictionary into a mapping of word->frequency."""
    with open(dictionary_path, 'r') as f:
        for line in f:
            word, freq = line.strip().split(' ')
            freq_dict[word] = int(freq)

    return freq_dict


freq_dict = {}

freq_dict = read_freq_dict(os.path.join(DICTS_DIR, 'diakorp.txt'), freq_dict)
freq_dict = read_freq_dict(os.path.join(DICTS_DIR, 'syn2020.txt'), freq_dict)
freq_dict = read_freq_dict(os.path.join(DICTS_DIR, 'all_words.txt'), freq_dict)


def get_frequency_score(text, freq_dict):
    """Compute a frequency score for a text using a freq dictionary."""
    score = 0

    words = text.split()
    for word in words:

        clean_word = word.lower().strip(".,;:!?()[]{}\"'")
        score += freq_dict.get(clean_word, 0)
    return score


def self_consistency_correction(lines_info, retries=3):
    """Call model ensemble and combine results into a self-consistent result."""
    corrections_list = []
    for model in models_list:
        for attempt in range(retries):
            try:
                # at least one API key is provided
                OR_API_key = API_KEYS.get('OPENROUTER_API_KEY', None)
                gemini_API_key = API_KEYS.get('GEMINI_API_KEY', None)

                result = None

                if model.startswith('gemini') and \
                    gemini_API_key is not None and \
                        not gemini_API_key.startswith('your_gemini_key_here'):
                    result = call_gemini_models(
                        lines_info, gemini_API_key, model=model)
                elif not model.startswith('gemini') and \
                    OR_API_key is not None and \
                        not OR_API_key.startswith('your_openrouter_key_here'):
                    result = call_llm(lines_info, OR_API_key, model=model)

                if result is not None:
                    if not isinstance(result, list):
                        corrections_list.append(result.get("textlines", []))
                    else:
                        corrections_list.append(result)
                break
            except Exception as e:
                logging.error(f"Error: {e}")
                # rate limit or other issues prevention
                if attempt != retries - 1:
                    logging.info(f"Retrying {model}")
                    time.sleep(10)

    # Build a mapping: textline id -> list of corrected texts.
    corrections_by_id = {}
    for corrections in corrections_list:
        for item in corrections:
            tid = item.get("id")
            corrected = item.get("corrected_text", "")
            if tid:
                corrections_by_id.setdefault(tid, []).append(corrected)

    # Create a mapping from textline id to the original text.
    original_by_id = {item["id"]: item["text"] for item in lines_info}

    final_corrections = {}
    for tid, texts in corrections_by_id.items():
        counter = Counter(texts)
        most_common_text, count = counter.most_common(1)[0]

        # if at least two models agree
        if count >= 2:
            final_candidate = most_common_text
        else:
            original_text = original_by_id.get(tid, "")
            # For each candidate, compute the similarity to the original text.
            final_candidate = None
            best_ed = float('inf')
            # Find the best matching candidate
            for candidate in texts:
                ed = pylev.levenshtein(candidate, original_text)
                if ed < best_ed:
                    best_ed = ed
                    final_candidate = candidate

        original_text = original_by_id.get(tid, "")
        candidate_score = get_frequency_score(final_candidate, freq_dict)
        original_score = get_frequency_score(original_text, freq_dict)

        # If the candidate's frequency score is not greater than the original's, keep the original.
        if candidate_score <= original_score:
            final_corrections[tid] = original_text
        else:
            final_corrections[tid] = final_candidate

    output = [{"id": tid, "corrected_text": final_corrections[tid]}
              for tid in final_corrections]
    return output


def load_models(region_model_path, line_model_path, recog_model_path, device='cuda:0'):
    """
    Load and return segmentation and recognition models.
    """
    if not region_model_path or line_model_path is None or recog_model_path is None:
        raise Exception('Some models were not specified')

    logging.info(f"Loading region segmentation model from {region_model_path}")
    region_model = vgsl.TorchVGSLModel.load_model(region_model_path)
    logging.info("Region segmentation model loaded")

    logging.info(f"Loading line segmentation model from {line_model_path}")
    line_model = vgsl.TorchVGSLModel.load_model(line_model_path)
    logging.info("Line segmentation model loaded")

    if recog_model_path.endswith('.mlmodel'):
        logging.info(
            f"Loading kraken text recognition model from {recog_model_path}")
        recog_model = models.load_any(recog_model_path, device)
        processor = None
        logging.info("Text recognition model loaded")
    else:
        logging.info(
            f"Loading TrOCR text recognition model from {recog_model_path}")
        processor = TrOCRProcessor.from_pretrained(recog_model_path)
        recog_model = VisionEncoderDecoderModel.from_pretrained(
            recog_model_path).to(device)

    return region_model, line_model, recog_model, processor


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
        default='pages/images',
        help='Path to the input folder.'
    )

    parser.add_argument(
        '--output-folder',
        type=str,
        default='pages/recognition_results',
        help='Path to the input file.'
    )

    parser.add_argument(
        '--text-line-segmentation-model',
        type=str,
        default='recognition/models/model_lines.mlmodel',
        help='Path to the text line segmentation model.'
    )

    parser.add_argument(
        '--text-region-segmentation-model',
        type=str,
        default='recognition/models/model_regions.mlmodel',
        help='Path to the text region segmentation model.'
    )

    parser.add_argument(
        '--text-recognition-model',
        type=str,
        default='recognition/models/model_recognition.mlmodel',
        help='Path to the text recognition model. Can be either kraken .mlmodel or TrOCR model folder'
    )

    if len(sys.argv) <= 1:
        parser.print_help()

    return parser.parse_args()


def process_file(file, record_type, output_folder, region_model, line_model, recog_model, processor, post_processing=False):
    """Process a single image file through segmentation, recognition and optional post-processing."""
    copied = False

    tmp_image_path = os.path.join(output_folder, os.path.basename(file))
    basename = os.path.splitext(os.path.basename(file))[0]

    # Define output paths
    region_xml = os.path.join(
        output_folder, f"{record_type}_{basename}_regions_segmented.xml")
    line_xml = os.path.join(
        output_folder, f"{record_type}_{basename}_lines_segmented.xml")
    seg_xml = os.path.join(
        output_folder, f"{record_type}_{basename}_segmented.xml")
    ocr_xml = os.path.join(output_folder, f"{record_type}_{basename}_ocr.xml")
    pp_xml = os.path.join(output_folder, f"{record_type}_{basename}_pp.xml")

    os.makedirs(output_folder, exist_ok=True)

    if file != tmp_image_path and not os.path.isfile(seg_xml) and not os.path.isfile(ocr_xml):
        shutil.copy(file, tmp_image_path)
        copied = True

    # if region assigned xml already exists, skip both region and line segmentation
    if not os.path.isfile(seg_xml):
        segment_page_into_lines(file, region_model, file, region_xml)

        segment_page_into_lines(file, line_model, file, line_xml)

        process_pagexml(region_xml, line_xml, seg_xml)
    else:
        logging.info(
            f"Segmentation for {file} already exists; skipping region and line segmentation.")

    if processor is None:
        # Use kraken for recognition
        infer_page(recog_model, file, seg_xml, ocr_xml)
    else:
        # otherwise use TrOCR
        recognize_text_into_pagexml_trocr(
            seg_xml, ocr_xml, Image.open(file), recog_model, processor)

    if post_processing:
        post_process_page_xml(ocr_xml,
                              pp_xml)

    if copied:
        os.remove(tmp_image_path)


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.CRITICAL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.input_folder is None:
        logging.error('No image directory provided')
        return

    # Verify folder structure
    expected = {'birth', 'death', 'marriage'}
    found = {name for name in os.listdir(args.input_folder) if os.path.isdir(
        os.path.join(args.input_folder, name))}
    if expected.intersection(found) != expected:
        logging.error(
            f"Input folder must contain subfolders: {expected}")
        return

    if args.output_folder is None:
        logging.warning(
            'No output path provided, output will be written to the same directory as the input file')

    # Verify post processing requirements
    post_process = args.post_processing
    OR_API_KEY = API_KEYS.get('OPENROUTER_API_KEY', None)
    GEMINI_API_KEY = API_KEYS.get('GEMINI_API_KEY', None)
    if args.post_processing and \
            (OR_API_KEY is None or OR_API_KEY.startswith('your_openrouter_key_here')) and \
            (GEMINI_API_KEY is None or GEMINI_API_KEY.startswith('your_gemini_key_here')):
        logging.warning(
            'No API key provided for post-processing. At least one key must be provided.\nSkipping post-processing.')
        post_process = False

    region_model, line_model, recog_model, processor = load_models(
        args.text_region_segmentation_model,
        args.text_line_segmentation_model,
        args.text_recognition_model
    )

    output_folder = args.output_folder if args.output_folder is not None else args.input_folder

    # process all images
    for record_type in os.listdir(args.input_folder):
        subdir = os.path.join(args.input_folder, record_type)
        if not os.path.isdir(subdir) or record_type not in {'birth', 'death', 'marriage'}:
            logging.warning(
                f"Skipping {record_type} as it is not a valid record type or not a directory.")
            continue

        for file in os.listdir(subdir):
            if file.endswith((".jpg", ".jpeg", ".png")):
                logging.info(f"Processing file: {file}")
                process_file(
                    os.path.join(subdir, file),
                    record_type,
                    output_folder,
                    region_model,
                    line_model,
                    recog_model,
                    processor,
                    post_processing=post_process
                )
                logging.info(
                    f"Processing of {file} completed successfully!\n" + "-"*40)


if __name__ == '__main__':
    main()
