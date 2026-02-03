# Historical Records Archive Application

This project contains an end-to-end pipeline for

1. Layout analysis of scanned vital records (birth/death/marriage)
2. Handwritten text recognition → PageXML outputs
3. Structured extraction to JSON (fields: name, date, place, relations, etc.)
4. Indexing and full-text search + field search
5. Flask web UI to browse records and check transcriptions against original images

---

## Project structure (abridged)

```
.
├── app/                               # Flask web app (static, templates)
├── pages/
│   ├── images/                        # original scanned images (birth|death|marriage)
│   ├── recognition_results/           # PageXML outputs from HTR
│   └── structured_records/            # JSON outputs after extraction
│   └── genealogy_structured_records/  # JSON outputs containing genealogical information
├── extraction/                        # structured_records_extraction.py (LLM-based)
├── genealogy/                         # genealogy.py
├── recognition/                       # HTR models, dictionaries & pipeline
├── .env.example
├── docker-compose.yml
├── process_images.sh
├── readme.md
├── requirements.txt
└── utils.py
```

---

## Prerequisites

- [Docker Desktop](https://docs.docker.com/desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/)

---

## Quick Start with Docker

1. **Navigate to the root folder**

   ```bash
   cd historical-archive-app
   ```

2. **Create a `.env` file with your API keys**.

   Copy `.env.example` to `.env` and fill in the keys. Example:

   ```bash
   cp .env.example .env
   # then edit .env and set your API keys
   ```

   _NOTE_: Docker reads environment variables when the container starts. If you edit `.env` while the app is running, run the following to apply changes and recreate containers with new variables: `docker-compose up -d`

3. **Build and run**

   ```bash
   docker-compose build base && docker-compose up -d --build
   ```

   This step will install necessary dependencies to run the project. It may take a while.

4. **Open** your browser at:

   > [http://localhost:5000](http://localhost:5000)

---

## Managing Docker Services for Specific Tasks

`docker-compose.yml` defines three services: `base`, `webapp`, and `ocr-worker`. You can control them individually or all at once.

### Starting and Stopping All Services

```bash
# Build and start all services
docker-compose build base && docker-compose up --build

# Stop running services without removing them
docker-compose stop

# Stop and remove all containers
docker-compose down
```

### Web application only

The webapp service runs the Flask web application.
View results at http://localhost:5000

```bash
# To start only the web application service:
docker-compose up -d webapp

# To stop only the web application service:
docker-compose stop webapp
```

### Pipeline processing only

```bash
# To start only the OCR worker service:
docker-compose up -d ocr-worker

# To stop only the OCR worker service:
docker-compose stop ocr-worker
```

## Processing Pipeline

There are two main steps to convert raw images into JSON and indexable records.

### 1. Text Recognition (Images → PageXML)

This step extracts text and text regions from the input images, resulting in a set of `.xml` files in PageXML format. It is crucial for correct information extraction later on.

- Place images (from e.g. https://ebadatelna.soapraha.cz/d/9548/129) in subfolders of folder `pages/images` based on their record type, i.e. `birth/`, `death/`, `marriage/`.
- To run the HTR pipeline, navigate to the `historical-archive-app` directory and execute the following:

```bash
# Ensure OCR worker is running
docker-compose up -d ocr-worker

# Inside Docker or from CLI:
docker-compose exec ocr-worker python recognition/pipeline.py
```

### 2. Structured Extraction (PageXML → JSON)

This step uses LLM to extract semantic information contained in recognized text from previous step in each text region.

- Ensure the OCR worker is running and that you are in `historical-archive-app directory`; then execute the extraction script:

```bash
# Inside Docker or from CLI:
docker-compose exec ocr-worker python extraction/structured_records_extraction.py
```

_Tip: For additional help, run either of the above commands with the `-h` switch. Mainly, you may want to try both kraken and TrOCR for text recognition by providing the respective models with `--text-recognition-model`. Fine-tuned CRNN models are available in this repository, fine-tuned Transformer models are available on Hugging Face: [Base model](https://huggingface.co/RPalk/trocr_fine_tuned_htr_base) and [Large model](https://huggingface.co/RPalk/trocr_fine_tuned_htr_large). Furthermore, you can turn on the post processing for text recognition (step 1) with `--post-processing` and/or try different OpenRouter models for information extraction (step 2) with `--model` since the availability of models changes over time._

### 3. Indexing the Records (JSON → Searchable UI)

After this pipeline, the Flask app will display and let you search through extracted vital records.
You can, and should, reload the extracted information directly in the app by providing the path to structured records.

- Open the UI, click on the Build Index button in top right corner.

- Enter the path to your records JSON files and click the Build/Rebuild Index button.

---

## Shell Script

For convenience, the `process_images.sh` script runs the recognition and extraction steps in sequence.

1. Make the script executable:

```bash
chmod +x process_images.sh
```

2. Run the script:

```bash
./process_images.sh
```

---

## API Keys

### OpenRouter

1. Sign up at [OpenRouter](https://openrouter.ai/).
2. Retrieve your API key under **Settings → API Keys** (https://openrouter.ai/settings/keys).
3. Store it in `.env`:
   ```bash
   OPENROUTER_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

### Gemini (Optional)

1. Obtain a Gemini API key from [AI Studio](https://aistudio.google.com/apikey).
2. Store it in `.env` as `GEMINI_API_KEY` if you wish to use Gemini models.

---

## Using the Web Application

**Ensure the web application is running:**

```bash
docker-compose up -d webapp
```

### Search Records Tab

- **Record Type**: Select from All Records, Marriage, Birth, or Death records to filter your search.

- **Search Type**: Choose your search method:

  - **General Search**: Searches across all text fields
  - **Field-Specific Search**: Searches only specified field in the next drop-down menu
  - **Place Search**: Searches specifically in location/place fields
  - **Name Search**: Searches in name fields only
  - **Date Search**: Searches in fields associated with dates

- **Search Query**: Enter your search term (e.g., names, places, dates); partial matches are supported.

- **Result Limit**: Set maximum number of results to display (default: 10, unlimited: 0).

### Build Index

Use the button in upper right corner to create or update the search index after processing new records:

1. Click on the **Build Index**
2. Provide the path to your structured records directory (default: `pages/genealogy_structured_records/`)
3. (Optional) Check **Force rebuild genealogical information** to re-analyze all family connections.

**Note:** When you build the index for the first time, the application performs initial I/O intensive operations and analyzes records to create genealogical links, saving these enriched records to the pages/genealogy_structured_records/ directory. Subsequents builds can be faster, as the image snippets can already be saved.

### Viewing Records

Each search result is an expandable card that displays:

- **Structured data**: Names, dates, places, relationships, other data, and available genealogical information
- **Original Record Image**: Relevant scanned document image, serving as the baseline information
- **Transcribed Text View**: A toggle to overlay the HTR-recognized text on the image for verification.

---

## Citation

```
@misc{palkovic2025htr,
      AUTHOR = {Palkovič, Radoslav},
      TITLE = {Automated Transcription and Search in Historical Records Using Handwritten Text Recognition},
      YEAR = {2025},
      TYPE = {Master Thesis},
      INSTITUTION = {Masaryk University, Faculty of Informatics},
      LOCATION = {Brno},
      SUPERVISOR = {Michal Batko}
}
```
