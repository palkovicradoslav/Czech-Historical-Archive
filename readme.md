# Czech Historical Archive — AI pipeline

This repository is reorganized to follow a standard AI project layout.

Top-level structure now:

- `.github/` — CI workflows
- `config/` — config files
- `data/` — dataset and generated artifacts (formerly `pages/`)
- `scripts/` — helper scripts for migrations and asset moves
- `src/` — application source code (Flask app, pipelines)
- `tests/` — pytest tests
- `Dockerfile*`, `docker-compose.yml`, `requirements.txt`

Quick start (local development):

1. Create `.env` from `.env.example` and set API keys.

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY, GEMINI_API_KEY if used
```

2. (Optional) migrate legacy `pages/` into `data/`:

```bash
./scripts/migrate_pages_to_data.sh
```

3. Move recognition assets (models/dictionaries) into `src/`:

```bash
./scripts/move_recognition_assets.sh
```

4. Run app locally (dev):

```bash
python src/app/app.py
```

5. Run tests:

```bash
pip install -r requirements.txt pytest
pytest -q
```

Docker:

```bash
docker compose build --pull
docker compose up --build
```

CI: GitHub Actions runs lint + tests and builds a CI docker image.

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
