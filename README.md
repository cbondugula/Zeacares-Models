# ZeaCares NLP — AI Disease Surveillance Pipeline

> Automates disease classification, ICD-10 coding, and outbreak detection for Andhra Pradesh's IDSP program.

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url> zeacares-nlp && cd zeacares-nlp
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run model comparison (BioBERT vs ClinicalBERT vs PubMedBERT)
python src/models/model_comparator.py --input data/zeacares_upload_ready.csv.xlsx --sample 200

# 3. Run full classification pipeline
python src/pipeline/classifier.py --input data/zeacares_upload_ready.csv.xlsx --output results/

# 4. Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Deploy everything (Docker)
docker-compose up -d
```

## Project Structure

```
NLP/
├── docs/                     ← All documentation
│   ├── PROJECT_DOCUMENT.md   ← What, Why, What we got (layman terms)
│   ├── APPROACH.md           ← Model comparison & selection rationale
│   └── IMPLEMENTATION.md     ← Step-by-step build guide
├── src/
│   ├── data/
│   │   ├── preprocess.py     ← PII removal + structured extraction
│   │   └── icd10_mapper.py   ← ICD-10 / SNOMED CT mapping
│   ├── models/
│   │   ├── biobert_classifier.py
│   │   ├── clinicalbert_classifier.py  ← Primary model
│   │   ├── pubmedbert_classifier.py
│   │   └── model_comparator.py
│   ├── pipeline/
│   │   ├── ner_extractor.py
│   │   ├── classifier.py
│   │   └── trend_detector.py
│   └── api/
│       ├── main.py           ← FastAPI server
│       └── schemas.py
├── notebooks/
│   └── pipeline_demo.ipynb   ← End-to-end walkthrough
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Model Winner: ClinicalBERT

| Model | ICD-10 Accuracy | NER F1 | Speed |
|---|---|---|---|
| BioBERT | 82.1% | 0.834 | 245ms |
| **ClinicalBERT** | **91.4%** | **0.912** | **198ms** |
| PubMedBERT | 87.3% | 0.876 | 312ms |

ClinicalBERT wins because it was trained on MIMIC-III hospital notes — the same style as our PHC records.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/classify` | Classify a single clinical record |
| POST | `/classify/batch` | Process a CSV batch |
| GET | `/trends/{district}` | Disease trend data |
| GET | `/alerts/active` | Current outbreak alerts |
| GET | `/dashboard/summary` | Statewide statistics |
| POST | `/feedback` | Submit ICD code correction |

Interactive docs: http://localhost:8000/docs

## Requirements

- Python 3.9+
- 8GB RAM minimum (16GB recommended)
- GPU optional but recommended for speed
- PostgreSQL 14+

## Environment Variables

```bash
# .env
DATABASE_URL=postgresql://zeacares:password@localhost:5432/zeacares
OPENROUTER_API_KEY=your_key_here        
REDIS_URL=redis://localhost:6379
MODEL_CACHE_DIR=./model_cache
LOG_LEVEL=INFO
```

## License

Open-source — MIT License. Designed for Government of Andhra Pradesh IDSP deployment.
