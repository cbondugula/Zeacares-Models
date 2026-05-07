"""
ZeaCares FastAPI Backend
Serves the classification pipeline, trend data, and outbreak alerts via REST API.
Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection

from src.api.schemas import (
    ActiveAlertsResponse,
    AlertResponse,
    BatchClassificationResponse,
    ClassifyRequest,
    ClassificationResponse,
    DashboardSummary,
    FeedbackRequest,
    HealthCheckResponse,
    TrendDataPoint,
    TrendResponse,
)

logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
_classifier   = None
_trend_detector = None
_mongo_client = None
_db           = None

# Anchor results dir to the project root regardless of where PM2/uvicorn is invoked from
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(_PROJECT_ROOT / "results")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_col(name: str) -> Collection:
    return _db[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _trend_detector, _mongo_client, _db

    # ── MongoDB ───────────────────────────────────────────────────────────────
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    _mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    _db = _mongo_client["zeacares"]
    # Indexes for fast queries
    _db["classifications"].create_index([("district", ASCENDING), ("disease_category", ASCENDING)])
    _db["classifications"].create_index([("record_id", ASCENDING)])
    _db["classifications"].create_index([("created_at", DESCENDING)])
    logger.info("MongoDB connected → zeacares")

    # ── Classifier (loads ClinicalBERT once) ──────────────────────────────────
    logger.info("Loading ZeaCares classifier (ClinicalBERT)...")
    from src.pipeline.classifier import ZeaCaresClassifier
    from src.pipeline.trend_detector import TrendDetector
    # Resolve model cache to an absolute path so HuggingFace caches in the project
    _model_cache = os.getenv("MODEL_CACHE_DIR", str(_PROJECT_ROOT / "model_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", _model_cache)
    os.environ.setdefault("HF_HOME", _model_cache)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _model_cache)
    Path(_model_cache).mkdir(parents=True, exist_ok=True)

    _classifier = ZeaCaresClassifier(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        device=os.getenv("DEVICE", "cpu"),
        model_cache_dir=_model_cache,
    )
    _classifier._init()   # pre-load model at startup — not per request
    _trend_detector = TrendDetector()
    logger.info("ZeaCares API ready")
    yield

    _mongo_client.close()
    logger.info("Shutting down ZeaCares API")


app = FastAPI(
    title="ZeaCares Disease Surveillance API",
    description=(
        "AI-powered disease classification and outbreak detection for "
        "Andhra Pradesh IDSP. Classifies clinical records to ICD-10/SNOMED CT "
        "and detects outbreak signals in real time."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    db_ok = False
    try:
        _mongo_client.admin.command("ping")
        db_ok = True
    except Exception:
        pass
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=_classifier is not None and _classifier._initialized,
        database_connected=db_ok,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ── Classification ─────────────────────────────────────────────────────────────

@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_single(request: ClassifyRequest):
    """Classify a single clinical text record and store result in MongoDB."""
    if not _classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    try:
        result = _classifier.classify(
            clinical_text=request.clinical_text,
            district=request.district,
            record_id=request.record_id or "api_call",
        )
        doc = asdict(result)
        doc["created_at"] = datetime.utcnow()
        doc["source"] = "api"
        get_col("classifications").insert_one(doc)
        doc.pop("_id", None)
        return ClassificationResponse(**result.__dict__)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=BatchClassificationResponse, tags=["Classification"])
async def classify_batch(
    file: UploadFile = File(..., description="CSV or XLSX file with clinicalText, district columns"),
):
    """Process a batch file synchronously. Returns real counts. Stores all results in MongoDB."""
    if not _classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    if not file.filename.endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx files accepted")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = str(RESULTS_DIR / f"classified_{timestamp}.json")
    canonical_path = str(RESULTS_DIR / "classified.json")

    suffix = ".xlsx" if file.filename.endswith(".xlsx") else ".csv"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Run synchronously so we can return real counts
        import asyncio
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, _classifier.classify_batch, tmp_path, output_path
        )

        if not results:
            raise HTTPException(status_code=422, detail="No records could be parsed from the file")

        # Write canonical path for trend/alert endpoints
        import shutil
        shutil.copy(output_path, canonical_path)

        # Build stats
        processed = sum(1 for r in results if r.icd10_code != "R69" or r.confidence > 0)
        failed = len(results) - processed
        sources: dict = {}
        review_count = 0
        total_ms = 0.0
        docs = []
        for r in results:
            sources[r.classification_source] = sources.get(r.classification_source, 0) + 1
            if r.review_required:
                review_count += 1
            total_ms += r.processing_time_ms
            doc = asdict(r)
            doc["created_at"] = datetime.utcnow()
            doc["source"] = "batch"
            doc["batch_file"] = file.filename
            doc["batch_ts"] = timestamp
            docs.append(doc)

        # Store all in MongoDB
        if docs:
            get_col("classifications").insert_many(docs)
            logger.info(f"Stored {len(docs)} records in MongoDB zeacares.classifications")

        return BatchClassificationResponse(
            total_records=len(results),
            processed=processed,
            failed=failed,
            avg_processing_time_ms=round(total_ms / len(results), 2) if results else 0.0,
            classification_sources=sources,
            review_required_count=review_count,
            results_path=output_path,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── Trends ─────────────────────────────────────────────────────────────────────

@app.get("/trends/{district}", response_model=TrendResponse, tags=["Surveillance"])
async def get_trends(
    district: str,
    disease_category: Optional[str] = Query(default=None),
    days: int = Query(default=30, ge=7, le=365),
):
    """Get disease trend data from MongoDB for a specific AP district."""
    import pandas as pd

    query: dict = {"district": district}
    if disease_category:
        query["disease_category"] = disease_category

    count = get_col("classifications").count_documents(query)
    if count == 0:
        raise HTTPException(status_code=404, detail=f"No records found for district: {district}")

    cursor = get_col("classifications").find(query, {"_id": 0})
    df = pd.DataFrame(list(cursor))

    if "created_at" not in df.columns:
        df["created_at"] = datetime.utcnow()
    df["date"] = pd.to_datetime(df["created_at"]).dt.normalize()

    series = df.groupby("date").size().rename("case_count")
    full_range = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_range, fill_value=0)

    try:
        cusum_scores, _ = _trend_detector.cusum.fit_predict(series)
        _, prophet_anomaly = _trend_detector.prophet.detect_anomalies(series)
    except Exception:
        cusum_scores, prophet_anomaly = {}, {}

    data_points = [
        TrendDataPoint(
            date=str(d.date()),
            case_count=int(v),
            cusum_score=round(float(cusum_scores.get(d, 0.0)), 2),
            is_anomaly=bool(prophet_anomaly.get(d, False)),
        )
        for d, v in series.items()
    ]

    return TrendResponse(
        district=district,
        disease_category=disease_category or "All",
        date_range=f"{series.index.min().date()} to {series.index.max().date()}",
        total_cases=int(series.sum()),
        daily_avg=round(float(series.mean()), 1),
        peak_day=str(series.idxmax().date()) if len(series) > 0 else None,
        peak_count=int(series.max()),
        data_points=data_points,
        has_alert=any(dp.is_anomaly or dp.cusum_score > 5 for dp in data_points[-7:]),
    )


# ── Alerts ─────────────────────────────────────────────────────────────────────

@app.get("/alerts/active", response_model=ActiveAlertsResponse, tags=["Surveillance"])
async def get_active_alerts():
    """Get active outbreak alerts from MongoDB."""
    alerts = list(get_col("alerts").find({}, {"_id": 0}).sort("triggered_at", DESCENDING))
    severity_counts = {s: sum(1 for a in alerts if a.get("alert_severity") == s)
                       for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    return ActiveAlertsResponse(
        total_alerts=len(alerts),
        critical=severity_counts["CRITICAL"],
        high=severity_counts["HIGH"],
        medium=severity_counts["MEDIUM"],
        low=severity_counts["LOW"],
        alerts=[AlertResponse(**a) for a in alerts],
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/alerts/refresh", tags=["Surveillance"])
async def refresh_alerts(background_tasks: BackgroundTasks):
    """Trigger alert re-detection from MongoDB data."""
    count = get_col("classifications").count_documents({})
    if count == 0:
        raise HTTPException(status_code=404, detail="No classified records in database")

    def _run():
        classified_path = RESULTS_DIR / "classified.json"
        alerts_path = RESULTS_DIR / "alerts.json"
        if classified_path.exists():
            _trend_detector.run(str(classified_path), str(alerts_path))
            if alerts_path.exists():
                with open(alerts_path) as f:
                    data = json.load(f)
                alerts = data.get("alerts", [])
                if alerts:
                    get_col("alerts").delete_many({})
                    for a in alerts:
                        a["refreshed_at"] = datetime.utcnow()
                    get_col("alerts").insert_many(alerts)
                    logger.info(f"Stored {len(alerts)} alerts in MongoDB")

    background_tasks.add_task(_run)
    return {"status": "refreshing", "message": "Alert refresh started"}


# ── Dashboard ──────────────────────────────────────────────────────────────────

@app.get("/dashboard/summary", response_model=DashboardSummary, tags=["Dashboard"])
async def dashboard_summary():
    """Statewide summary from MongoDB."""
    import pandas as pd

    count = get_col("classifications").count_documents({})
    if count == 0:
        raise HTTPException(status_code=404, detail="No classified records found. Run /classify/batch first.")

    cursor = get_col("classifications").find({}, {"_id": 0,
        "created_at": 1, "disease_raw": 1, "disease_category": 1, "district": 1})
    df = pd.DataFrame(list(cursor))

    now = datetime.utcnow()
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        records_7d  = int(df[df["created_at"] >= now - pd.Timedelta(days=7)].shape[0])
        records_30d = int(df[df["created_at"] >= now - pd.Timedelta(days=30)].shape[0])
    else:
        records_7d = records_30d = count

    top_diseases = (
        df["disease_raw"].value_counts().head(10)
        .reset_index()
        .rename(columns={"disease_raw": "disease", "count": "count"})
        .to_dict("records")
    )

    active_alerts = get_col("alerts").count_documents({})

    return DashboardSummary(
        total_records_7d=records_7d,
        total_records_30d=records_30d,
        active_alerts=active_alerts,
        districts_reporting=int(df["district"].nunique()),
        top_diseases=top_diseases,
        category_breakdown=df["disease_category"].value_counts().to_dict(),
        coverage_by_district=df.groupby("district").size().to_dict(),
        last_updated=now.isoformat() + "Z",
    )


# ── Feedback ───────────────────────────────────────────────────────────────────

@app.post("/feedback", tags=["Improvement"])
async def submit_feedback(request: FeedbackRequest):
    """Store ICD correction in MongoDB for model fine-tuning."""
    doc = {**request.model_dump(), "submitted_at": datetime.utcnow()}
    get_col("feedback").insert_one(doc)
    total = get_col("feedback").count_documents({})
    return {
        "status": "accepted",
        "total_feedback_records": total,
        "message": "Correction recorded. Will be included in next model fine-tuning cycle.",
    }


@app.get("/feedback/stats", tags=["Improvement"])
async def feedback_stats():
    total = get_col("feedback").count_documents({})
    return {
        "total": total,
        "ready_for_finetuning": total >= 50,
        "threshold": 50,
        "message": f"{total}/50 corrections collected for next fine-tuning cycle",
    }


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
