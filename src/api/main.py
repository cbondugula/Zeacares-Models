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
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# Global classifier instance (loaded once at startup)
_classifier = None
_trend_detector = None

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _trend_detector
    logger.info("Loading ZeaCares classifier...")
    from src.pipeline.classifier import ZeaCaresClassifier
    from src.pipeline.trend_detector import TrendDetector

    _classifier = ZeaCaresClassifier(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        device=os.getenv("DEVICE"),
    )
    _trend_detector = TrendDetector()
    logger.info("ZeaCares API ready")
    yield
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
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=_classifier is not None,
        database_connected=True,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ── Classification ─────────────────────────────────────────────────────────────

@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_single(request: ClassifyRequest):
    """
    Classify a single clinical text record.
    Returns ICD-10 code, disease category, confidence, and review flag.
    Processing time: ~200ms (lookup/embedding) to ~3s (LLM fallback).
    """
    if not _classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    try:
        result = _classifier.classify(
            clinical_text=request.clinical_text,
            district=request.district,
            record_id=request.record_id or "api_call",
        )
        return ClassificationResponse(**result.__dict__)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=BatchClassificationResponse, tags=["Classification"])
async def classify_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV or XLSX file with clinicalText, district columns"),
):
    """
    Process a batch CSV/XLSX file. Returns immediately with a job ID.
    Results are saved to results/classified_{timestamp}.json.
    """
    if not _classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    if not file.filename.endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx files accepted")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = str(RESULTS_DIR / f"classified_{timestamp}.json")

    # Save uploaded file
    suffix = ".xlsx" if file.filename.endswith(".xlsx") else ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    def _run_batch():
        try:
            _classifier.classify_batch(tmp_path, output_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    background_tasks.add_task(_run_batch)

    return BatchClassificationResponse(
        total_records=0,
        processed=0,
        failed=0,
        avg_processing_time_ms=0.0,
        classification_sources={},
        review_required_count=0,
        results_path=output_path,
    )


# ── Trends ─────────────────────────────────────────────────────────────────────

@app.get("/trends/{district}", response_model=TrendResponse, tags=["Surveillance"])
async def get_trends(
    district: str,
    disease_category: Optional[str] = Query(default=None),
    days: int = Query(default=30, ge=7, le=365),
):
    """
    Get disease trend data for a specific AP district.
    Returns daily case counts with CUSUM scores for outbreak monitoring.
    """
    classified_path = RESULTS_DIR / "classified.json"
    if not classified_path.exists():
        raise HTTPException(status_code=404,
                            detail="No classified records found. Run /classify/batch first.")
    try:
        df = _trend_detector.load_classified_records(str(classified_path))
        df = df[df["district"] == district]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No records found for district: {district}")

        if disease_category:
            df = df[df["disease_category"] == disease_category]

        agg = _trend_detector.aggregate_daily(df)
        group = agg[agg["district"] == district]
        if disease_category:
            group = group[group["disease_category"] == disease_category]

        import pandas as pd
        full_range = pd.date_range(
            group["date"].min(),
            group["date"].max(),
            freq="D"
        )
        series = (
            group.groupby("date")["case_count"]
            .sum()
            .reindex(full_range, fill_value=0)
        )
        cusum_scores, _ = _trend_detector.cusum.fit_predict(series)
        _, prophet_anomaly = _trend_detector.prophet.detect_anomalies(series)

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trend query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Alerts ─────────────────────────────────────────────────────────────────────

@app.get("/alerts/active", response_model=ActiveAlertsResponse, tags=["Surveillance"])
async def get_active_alerts():
    """
    Get all currently active outbreak alerts across AP.
    Sorted by severity: CRITICAL → HIGH → MEDIUM → LOW.
    """
    alerts_path = RESULTS_DIR / "alerts.json"
    if not alerts_path.exists():
        return ActiveAlertsResponse(
            total_alerts=0, critical=0, high=0, medium=0, low=0,
            alerts=[], generated_at=datetime.utcnow().isoformat() + "Z",
        )

    with open(alerts_path) as f:
        data = json.load(f)

    alerts = [AlertResponse(**a) for a in data.get("alerts", [])]
    severity_counts = {s: 0 for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    for a in alerts:
        severity_counts[a.alert_severity] = severity_counts.get(a.alert_severity, 0) + 1

    return ActiveAlertsResponse(
        total_alerts=len(alerts),
        critical=severity_counts["CRITICAL"],
        high=severity_counts["HIGH"],
        medium=severity_counts["MEDIUM"],
        low=severity_counts["LOW"],
        alerts=alerts,
        generated_at=data.get("generated_at", datetime.utcnow().isoformat() + "Z"),
    )


@app.post("/alerts/refresh", tags=["Surveillance"])
async def refresh_alerts(background_tasks: BackgroundTasks):
    """Trigger a background re-run of trend detection and update alerts.json."""
    classified_path = RESULTS_DIR / "classified.json"
    if not classified_path.exists():
        raise HTTPException(status_code=404, detail="No classified records found")

    def _run():
        _trend_detector.run(str(classified_path), str(RESULTS_DIR / "alerts.json"))

    background_tasks.add_task(_run)
    return {"status": "refreshing", "message": "Alert refresh started in background"}


# ── Dashboard ──────────────────────────────────────────────────────────────────

@app.get("/dashboard/summary", response_model=DashboardSummary, tags=["Dashboard"])
async def dashboard_summary():
    """
    Statewide summary statistics for the main dashboard.
    Returns: total records, active alerts, district coverage, top diseases.
    """
    classified_path = RESULTS_DIR / "classified.json"
    if not classified_path.exists():
        raise HTTPException(status_code=404, detail="No classified records found")

    with open(classified_path) as f:
        records = json.load(f)

    import pandas as pd
    df = pd.DataFrame(records)

    now = datetime.utcnow()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        records_7d = int(df[df["date"] >= now - pd.Timedelta(days=7)].shape[0])
        records_30d = int(df[df["date"] >= now - pd.Timedelta(days=30)].shape[0])
    else:
        records_7d = len(records)
        records_30d = len(records)

    top_diseases = (
        df["disease_raw"].value_counts().head(10)
        .reset_index().rename(columns={"index": "disease", "disease_raw": "count"})
        .to_dict("records")
    )

    alerts_path = RESULTS_DIR / "alerts.json"
    active_alerts = 0
    if alerts_path.exists():
        with open(alerts_path) as f:
            active_alerts = json.load(f).get("total_alerts", 0)

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


# ── Feedback / Model Improvement Loop ─────────────────────────────────────────

@app.post("/feedback", tags=["Improvement"])
async def submit_feedback(request: FeedbackRequest):
    """
    Medical officer submits ICD code correction for a misclassified record.
    Corrections are stored and used for weekly ClinicalBERT fine-tuning.
    """
    feedback_path = RESULTS_DIR / "feedback.json"
    existing = []
    if feedback_path.exists():
        with open(feedback_path) as f:
            existing = json.load(f)

    existing.append({
        **request.model_dump(),
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    })

    with open(feedback_path, "w") as f:
        json.dump(existing, f, indent=2)

    return {
        "status": "accepted",
        "total_feedback_records": len(existing),
        "message": "Correction recorded. Will be included in next model fine-tuning cycle.",
    }


@app.get("/feedback/stats", tags=["Improvement"])
async def feedback_stats():
    """Summary of collected corrections for fine-tuning readiness."""
    feedback_path = RESULTS_DIR / "feedback.json"
    if not feedback_path.exists():
        return {"total": 0, "ready_for_finetuning": False, "threshold": 50}

    with open(feedback_path) as f:
        records = json.load(f)

    return {
        "total": len(records),
        "ready_for_finetuning": len(records) >= 50,
        "threshold": 50,
        "message": f"{len(records)}/50 corrections collected for next fine-tuning cycle",
    }


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
