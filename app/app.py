import os
import pickle
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory, send_file
from records_indexer import VitalRecordsIndexer
import logging
import sys
import json
import xml.etree.ElementTree as ET
from io import BytesIO
from PIL import Image
from functools import lru_cache


sys.path.insert(1, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from utils import setup_logger  # NOQA

sys.path.insert(1, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'genealogy')))

from genealogy import process_and_save  # NOQA

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

RECORDS_DIR = os.path.normpath(
    os.path.join(BASE_DIR, '..', 'pages', 'structured_records')
)
GEN_RECORDS_DIR = os.path.normpath(
    os.path.join(BASE_DIR, '..', 'pages', 'genealogy_structured_records')
)
CROPPED_IMAGES_DIR = os.path.normpath(
    os.path.join(BASE_DIR, 'static', 'images')
)
INDEX_DIR = os.path.normpath(
    os.path.join(BASE_DIR, 'vital_records_index')
)
IMAGES_DIR = os.path.normpath(
    os.path.join(BASE_DIR, '..', 'pages', 'images')
)
STATE_FILE = os.path.join(INDEX_DIR, 'records_state.json')

GENEALOGY_FILE = os.path.join(INDEX_DIR, 'family_tree.pkl')

# Create static directories if they don't exist
if not os.path.exists("static"):
    os.makedirs("static")
if not os.path.exists(CROPPED_IMAGES_DIR):
    os.makedirs(CROPPED_IMAGES_DIR)

# Check if index exists before initializing
if os.path.exists(INDEX_DIR):
    indexer = VitalRecordsIndexer(index_dir=INDEX_DIR)
else:
    logging.warning(
        f"Index directory {INDEX_DIR} does not exist. Please build the index first.")
    indexer = None

family_tree_builder = None

# The following functions maintain cropped images
# If the structured records changed since last index build -> remove all cropped images


def load_records_state():
    """Load the saved state of records from a JSON file"""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_records_state(state):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def get_current_records_state():
    """Handles initialization of the records state"""
    state = {}
    for root, _, files in os.walk(RECORDS_DIR):
        for fname in files:
            full_path = os.path.join(root, fname)
            try:
                state[full_path] = os.path.getmtime(full_path)
            except OSError:
                continue
    return state


def clear_cropped_images_and_gen_records():
    """Remove all cropped images and genealogy structured records"""
    for fname in os.listdir(CROPPED_IMAGES_DIR):
        path = os.path.join(CROPPED_IMAGES_DIR, fname)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError as e:
                logging.warning(f"Failed to delete {path}: {e}")

    for fname in os.listdir(GEN_RECORDS_DIR):
        path = os.path.join(GEN_RECORDS_DIR, fname)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError as e:
                logging.warning(f"Failed to delete {path}: {e}")


def save_family_tree_builder(family_tree_builder):
    """Save the family tree builder to a pickle file"""
    try:
        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(GENEALOGY_FILE, 'wb') as f:
            pickle.dump(family_tree_builder, f)
        logging.info("Family tree builder saved to file")
    except Exception as e:
        logging.error(f"Error saving family tree builder: {str(e)}")


def load_family_tree_builder():
    """Load the family tree builder from a pickle file"""
    global family_tree_builder

    try:
        if os.path.exists(GENEALOGY_FILE):
            with open(GENEALOGY_FILE, 'rb') as f:
                family_tree_builder = pickle.load(f)
            logging.info("Family tree builder loaded from file")
            return family_tree_builder
        else:
            logging.info("No existing family tree file found")
            return None
    except Exception as e:
        logging.error(f"Error loading family tree builder: {str(e)}")
        return None


def get_genealogy_info(record):
    """Get genealogical information for a record"""
    if not family_tree_builder:
        return {}

    record_type = record.get('record_type', '')
    genealogy_info = {}

    if record_type == 'marriage':
        # For marriages, find children
        groom_id = record.get('groom_id')
        bride_id = record.get('bride_id')

        if groom_id is not None and bride_id is not None and groom_id != '' and bride_id != '':
            groom = family_tree_builder.people.get(int(groom_id))
            bride = family_tree_builder.people.get(int(bride_id))

            if groom and bride:
                # Find common children
                common_children = []
                for child in groom.children:
                    if child in bride.children:
                        common_children.append({
                            'name': child.full_name,
                            'birthdate': str(child.birthdate) if child.birthdate else None,
                            'birthplace': child.birthplace
                        })

                genealogy_info['children'] = common_children

    elif record_type == 'birth':
        # For births, find marriage records of the same person
        person_id = record.get('person_id')

        if person_id is not None and person_id != '':
            person = family_tree_builder.people.get(int(person_id))

            if person:
                marriages = []
                for spouse in person.spouses:
                    marriages.append({
                        'spouse_name': spouse.full_name,
                        'wedding_date': str(person.weddingdate) if person.weddingdate else None
                    })

                genealogy_info['marriages'] = marriages

                if person.deathdate and person.deathplace:
                    genealogy_info['death_info'] = {
                        'deathdate': str(person.deathdate),
                        'deathplace': person.deathplace
                    }

                children = []
                for child in person.children:
                    children.append({
                        'name': child.full_name,
                        'birthdate': str(child.birthdate) if child.birthdate else None
                    })

                genealogy_info['children'] = children

    elif record_type == 'death':
        # For deaths, find birth record of the same person
        person_id = record.get('person_id')

        if person_id is not None and person_id != '':
            person = family_tree_builder.people.get(int(person_id))

            if person:
                genealogy_info['birth_info'] = {
                    'birthdate': str(person.birthdate) if person.birthdate else None,
                    'birthplace': person.birthplace
                }

                # Add marriage information
                marriages = []
                for spouse in person.spouses:
                    marriages.append({
                        'spouse_name': spouse.full_name,
                        'wedding_date': str(person.weddingdate) if person.weddingdate else None
                    })

                genealogy_info['marriages'] = marriages

                # Add children
                children = []
                for child in person.children:
                    children.append({
                        'name': child.full_name,
                        'birthdate': str(child.birthdate) if child.birthdate else None
                    })

                genealogy_info['children'] = children

    return genealogy_info


def compute_bounding_box(all_coords, padding=0):
    """Calculates bounding box from a string of coordinates with optional padding"""
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


@app.route('/api/placeholder/<int:width>/<int:height>')
def placeholder(width, height):
    """Returns placeholder image for not loaded real images"""
    img = Image.new('RGB', (width, height), color=(220, 220, 220))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route('/')
def home():
    """Render the home page with search form"""
    return render_template('index.html')


@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """Serve images from the images directory"""
    return send_from_directory(CROPPED_IMAGES_DIR, filename)


@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    if indexer is None:
        return jsonify({"error": "Index not initialized. Please build the index first."}), 500

    search_type = request.form.get('search_type', 'general')
    query = request.form.get('query', '')
    field = request.form.get('field', None)
    record_type = request.form.get('record_type', None)
    limit = int(request.form.get('limit', 10))
    limit = None if limit == 0 else limit

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Early return for very short queries
    if len(query) < 2:
        return jsonify({"error": "Query too short"}), 400

    try:
        results = []
        if search_type == 'general':
            results = indexer.search(
                query, record_type=record_type, limit=limit)
        elif search_type == 'field' and field:
            results = indexer.search(
                query, field=field, record_type=record_type, limit=limit)
        elif search_type == 'name':
            results = indexer.search_by_name(
                query, record_type=record_type, limit=limit)
        elif search_type == 'place':
            results = indexer.search_by_place(
                query, record_type=record_type, limit=limit)
        elif search_type == 'date':
            results = indexer.search_by_date(
                query, record_type=record_type, limit=limit)

        processed_results = []
        for r in results:

            if r.get('cropped_image_url'):
                r['image_url'] = url_for(
                    'serve_image', filename=r['cropped_image_url'])
            else:
                # Fallback to a placeholder if image creation failed during indexing
                r['image_url'] = url_for('placeholder', width=300, height=100)

            # Clean up null values for display
            for k, v in r.items():
                if v == "null":
                    r[k] = 'Unknown'

            # Add genealogy information
            r['genealogy'] = get_genealogy_info(r)
            gen_bi = r['genealogy'].get('birth_info', {})
            if gen_bi != {} and ('born on' in r['additional_info'] or
                                 (gen_bi != {} and gen_bi.get('birthdate') is None and r.get('place_of_living') is not None)):
                r['genealogy']['birth_info'] = None

            processed_results.append(r)

        return jsonify({"results": processed_results, "count": len(processed_results)})

    except Exception as e:
        return jsonify({"error": f"Search error: {str(e)}"}), 500


@app.route('/build_index', methods=['POST'])
def build_index():
    """Build or rebuild the index from JSON data"""
    global indexer
    global family_tree_builder

    input_path = request.form.get(
        'input_path', '')
    if input_path == '':
        input_path = GEN_RECORDS_DIR if os.path.exists(
            GEN_RECORDS_DIR) else RECORDS_DIR
    force_rebuild_genealogy = request.form.get(
        'force_rebuild_genealogy') == 'on'

    if not os.path.exists(input_path):
        return jsonify({"error": f"Input path {input_path} does not exist"}), 400

    # Compare state
    old_state = load_records_state()
    new_state = get_current_records_state()

    if old_state != new_state:
        logging.info(
            "Detected changes in records directory; clearing cropped images.")
        clear_cropped_images_and_gen_records()
        save_records_state(new_state)
        force_rebuild_genealogy = True
    else:
        logging.info(
            "No changes in records directory; cropped images left intact.")

    try:
        # Handle genealogy data loading/rebuilding
        if force_rebuild_genealogy:
            logging.info(
                "Force rebuild genealogy option selected - recalculating genealogical connections")
            family_tree_builder = process_and_save(
                RECORDS_DIR, GEN_RECORDS_DIR)
            save_family_tree_builder(family_tree_builder)
            genealogy_message = "genealogical information recalculated"
        else:
            # Try to load existing genealogy data first
            family_tree_builder = load_family_tree_builder()
            if family_tree_builder is None:
                logging.info(
                    "No existing genealogy data found - calculating genealogical connections")
                family_tree_builder = process_and_save(
                    RECORDS_DIR, GEN_RECORDS_DIR)
                save_family_tree_builder(family_tree_builder)
                genealogy_message = "genealogical information calculated (no existing data found)"
            else:
                genealogy_message = "genealogical information loaded from file"

        indexer = VitalRecordsIndexer(index_dir=INDEX_DIR)
        record_count = indexer.build_index(input_path)

        return jsonify({
            "success": True,
            "message": f"Successfully indexed {record_count} vital records, {genealogy_message}",
            "record_count": record_count
        })
    except Exception as e:
        return jsonify({"error": f"Indexing error: {str(e)}"}), 500


@lru_cache(maxsize=1000)
def extract_text_lines_from_pagexml(pagexml_path, region_id):
    """"Returns text lines for specified text region"""
    try:
        tree = ET.parse(pagexml_path)
        root = tree.getroot()

        ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}

        text_region = root.find(f".//ns:TextRegion[@id='{region_id}']", ns)
        if text_region is None:
            return []

        text_lines = []
        for text_line_elem in text_region.findall(".//ns:TextLine", ns):
            text_elem = text_line_elem.find(".//ns:TextEquiv/ns:Unicode", ns)
            text = text_elem.text if text_elem is not None and text_elem.text is not None else ""

            # Get the line coordinates
            coords_elem = text_line_elem.find(".//ns:Coords", ns)
            points = coords_elem.get(
                'points') if coords_elem is not None else ""

            line_id = text_line_elem.get('id', '')

            # Add normalized information to the text_lines
            text_lines.append({
                'id': line_id,
                'text': text,
                'points': points
            })

        return text_lines

    except Exception as e:
        logging.error(f"Error extracting text lines: {str(e)}")
        return []


@app.route('/get_text_lines')
def get_text_lines():
    """Get text lines for a specific region"""
    region_id = request.args.get('region_id')
    file_path = request.args.get('file_path')

    if not region_id or not file_path:
        return jsonify({"error": "Missing region_id or file_path parameter"}), 400

    pagexml_path = file_path

    if not os.path.exists(pagexml_path):
        return jsonify({"error": f"PageXML file not found: {pagexml_path}"}), 404

    try:
        text_lines = extract_text_lines_from_pagexml(pagexml_path, region_id)

        region_coords = get_region_coordinates(pagexml_path, region_id)

        return jsonify({
            "text_lines": text_lines,
            "region_coords": region_coords
        })
    except Exception as e:
        return jsonify({"error": f"Error extracting text lines: {str(e)}"}), 500


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


if __name__ == '__main__':
    setup_logger()

    app.run(debug=False, port=5000, host='0.0.0.0')
