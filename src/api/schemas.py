"""
Pydantic schemas for ZeaCares FastAPI endpoints.
Defines request/response shapes for all API calls.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Request Schemas ────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    clinical_text: str = Field(..., min_length=10, description="Raw clinical text from PHC record")
    district: str = Field(default="Unknown", description="AP district name")
    record_id: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "clinical_text": "Female 60 years, presented with hypertension stage 2. Onset was gradual with duration of 3 day(s). Severity: mild. Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. BMI status: Healthy. Attended UPHC Lankapatnam.",
                "district": "Vizianagaram",
            }
        }
    }


class FeedbackRequest(BaseModel):
    record_id: str
    original_icd10_code: str
    corrected_icd10_code: str
    corrected_description: str
    corrected_category: str
    officer_id: str
    notes: Optional[str] = None


# ── Response Schemas ───────────────────────────────────────────────────────────

class ClassificationResponse(BaseModel):
    record_id: str
    disease_raw: Optional[str]
    disease_normalized: Optional[str]
    icd10_code: str
    icd10_description: str
    snomed_code: Optional[str]
    disease_category: str
    sub_category: str
    confidence: float
    classification_source: str
    review_required: bool
    processing_time_ms: float
    district: Optional[str]
    severity: Optional[str]
    age_band: Optional[str]
    gender: Optional[str]
    # Vitals
    temperature_f: Optional[float]
    pulse_bpm: Optional[int]
    bp_systolic: Optional[int]
    bp_diastolic: Optional[int]
    spo2_pct: Optional[float]
    bmi_status: Optional[str]
    facility: Optional[str]


class BatchClassificationResponse(BaseModel):
    total_records: int
    processed: int
    failed: int
    avg_processing_time_ms: float
    classification_sources: dict[str, int]
    review_required_count: int
    results_path: str


class TrendDataPoint(BaseModel):
    date: str
    case_count: int
    cusum_score: float
    is_anomaly: bool


class TrendResponse(BaseModel):
    district: str
    disease_category: str
    date_range: str
    total_cases: int
    daily_avg: float
    peak_day: Optional[str]
    peak_count: int
    data_points: list[TrendDataPoint]
    has_alert: bool


class AlertResponse(BaseModel):
    district: str
    disease_category: str
    alert_type: str
    alert_severity: str
    current_cases_7d: int
    expected_cases_7d: float
    cusum_score: float
    prophet_anomaly: bool
    percent_above_baseline: float
    triggered_at: str
    details: str


class ActiveAlertsResponse(BaseModel):
    total_alerts: int
    critical: int
    high: int
    medium: int
    low: int
    alerts: list[AlertResponse]
    generated_at: str


class DashboardSummary(BaseModel):
    total_records_7d: int
    total_records_30d: int
    active_alerts: int
    districts_reporting: int
    top_diseases: list[dict]
    category_breakdown: dict[str, int]
    coverage_by_district: dict[str, int]
    last_updated: str


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    database_connected: bool
    timestamp: str
