-- ZeaCares Database Schema
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS classified_records (
    id              SERIAL PRIMARY KEY,
    record_id       VARCHAR(64),
    disease_raw     TEXT,
    disease_normalized TEXT,
    icd10_code      VARCHAR(20),
    icd10_description TEXT,
    snomed_code     VARCHAR(20),
    disease_category VARCHAR(50),
    sub_category    VARCHAR(50),
    confidence      FLOAT,
    classification_source VARCHAR(20),
    review_required BOOLEAN DEFAULT FALSE,
    district        VARCHAR(100),
    severity        VARCHAR(20),
    age_band        VARCHAR(20),
    gender          CHAR(1),
    record_date     DATE DEFAULT CURRENT_DATE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_records_district ON classified_records(district);
CREATE INDEX idx_records_date ON classified_records(record_date);
CREATE INDEX idx_records_category ON classified_records(disease_category);
CREATE INDEX idx_records_icd10 ON classified_records(icd10_code);

CREATE TABLE IF NOT EXISTS outbreak_alerts (
    id              SERIAL PRIMARY KEY,
    district        VARCHAR(100),
    disease_category VARCHAR(50),
    alert_type      VARCHAR(30),
    alert_severity  VARCHAR(10),
    current_cases_7d INT,
    expected_cases_7d FLOAT,
    cusum_score     FLOAT,
    prophet_anomaly BOOLEAN,
    percent_above_baseline FLOAT,
    triggered_at    TIMESTAMPTZ DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,
    is_active       BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS officer_feedback (
    id              SERIAL PRIMARY KEY,
    record_id       VARCHAR(64),
    original_icd10_code VARCHAR(20),
    corrected_icd10_code VARCHAR(20),
    corrected_description TEXT,
    corrected_category VARCHAR(50),
    officer_id      VARCHAR(64),
    notes           TEXT,
    submitted_at    TIMESTAMPTZ DEFAULT NOW(),
    used_for_training BOOLEAN DEFAULT FALSE
);
