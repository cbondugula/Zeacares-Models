"""
End-to-end tests for ZeaCares NLP pipeline.
Tests: preprocessing, NER extraction, ICD-10 mapping, classification, trend detection.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Preprocessing Tests ────────────────────────────────────────────────────────

class TestPreprocessing:
    SAMPLE = "Female 60 years, presented with hypertension stage 2. Onset was gradual with duration of 3 day(s). Severity: mild. Vitals: Temperature 98F, Pulse 72bpm, BP 110/80mmHg, SpO2 98%. BMI status: Healthy. Attended UPHC Lankapatnam."

    def test_extracts_gender(self):
        from src.data.preprocess import extract_fields
        fields = extract_fields(self.SAMPLE)
        assert fields["gender"] == "F"

    def test_extracts_age(self):
        from src.data.preprocess import extract_fields
        fields = extract_fields(self.SAMPLE)
        assert fields["age"] == 60

    def test_extracts_disease(self):
        from src.data.preprocess import extract_fields
        fields = extract_fields(self.SAMPLE)
        assert fields["disease_raw"] == "hypertension stage 2"

    def test_extracts_bp(self):
        from src.data.preprocess import extract_fields
        fields = extract_fields(self.SAMPLE)
        assert fields["bp_sys"] == 110
        assert fields["bp_dia"] == 80

    def test_extracts_severity(self):
        from src.data.preprocess import extract_fields
        fields = extract_fields(self.SAMPLE)
        assert fields["severity"] == "mild"

    def test_anonymization_strips_age(self):
        from src.data.preprocess import anonymize_text
        clean = anonymize_text(self.SAMPLE, "F", 60)
        assert "60 years" not in clean
        assert "[F, 55-64]" in clean

    def test_age_to_band(self):
        from src.data.preprocess import _age_to_band
        assert _age_to_band(60) == "55-64"
        assert _age_to_band(0) == "0-14"
        assert _age_to_band(85) == "85-120"


# ── NER Tests ─────────────────────────────────────────────────────────────────

class TestNER:
    def setup_method(self):
        from src.pipeline.ner_extractor import NERExtractor
        self.extractor = NERExtractor(use_model=False)

    def test_extract_disease(self):
        text = "Male 45 years, presented with fever. Onset was sudden..."
        e = self.extractor.extract(text)
        assert e.disease_raw == "fever"

    def test_normalize_hypertension_variants(self):
        from src.pipeline.ner_extractor import normalize_disease
        assert normalize_disease("hypertension stage 2") == "hypertension"
        assert normalize_disease("stage 2 hypertension") == "hypertension"
        assert normalize_disease("HTN-2") == "hypertension"
        assert normalize_disease("high bp") == "hypertension"

    def test_normalize_diabetes_variants(self):
        from src.pipeline.ner_extractor import normalize_disease
        assert normalize_disease("diabetic on oral treatment") == "diabetes type 2"
        assert normalize_disease("diabetes monitored") == "diabetes type 2"

    def test_vitals_flags(self):
        text = "Male 60 years, presented with fever. Onset sudden. Severity: severe. Vitals: Temperature 103F, Pulse 110bpm, BP 150/95mmHg, SpO2 92%. BMI status: Overweight. Attended PHC."
        e = self.extractor.extract(text)
        vitals = self.extractor.get_vitals_summary(e)
        assert vitals["bp_risk"] == "high"
        assert vitals["fever"] is True
        assert vitals["low_spo2"] is True
        assert vitals["tachycardia"] is True


# ── ICD-10 Mapper Tests ────────────────────────────────────────────────────────

class TestICD10Mapper:
    def setup_method(self):
        from src.data.icd10_mapper import ICD10Mapper
        self.mapper = ICD10Mapper()

    def test_hypertension_maps_to_I10(self):
        r = self.mapper.map("hypertension stage 2")
        assert r.icd10_code == "I10"

    def test_variant_maps_correctly(self):
        r = self.mapper.map("stage 2 hypertension")
        assert r.icd10_code == "I10"

    def test_diabetes_maps_to_E11(self):
        r = self.mapper.map("diabetes monitored")
        assert r.icd10_code.startswith("E11")

    def test_fever_maps_to_R509(self):
        r = self.mapper.map("fever")
        assert r.icd10_code == "R50.9"

    def test_dog_bite_maps_correctly(self):
        r = self.mapper.map("dog bite")
        assert r.icd10_code == "W54.0"
        assert r.disease_category == "Communicable"
        assert r.sub_category == "Zoonotic"

    def test_unknown_disease_returns_unspecified(self):
        r = self.mapper.map("completely unknown disease xyz123")
        assert r.icd10_code == "R69"
        assert r.match_method == "unspecified"

    def test_coverage_of_known_conditions(self):
        known = ["fever", "hypertension stage 1", "diabetes monitored",
                 "cough", "backache", "dog bite", "allergy", "loose motion"]
        stats = self.mapper.coverage_stats(known)
        assert stats["coverage_pct"] == 100.0


# ── Classifier Tests (no model loading) ───────────────────────────────────────

class TestClassifierLookupPath:
    """Test the lookup/regex fast path — no model loading needed."""

    def test_classify_hypertension(self):
        from src.data.icd10_mapper import ICD10Mapper
        mapper = ICD10Mapper()
        r = mapper.map("hypertension stage 2")
        assert r.icd10_code == "I10"
        assert r.disease_category == "Non-Communicable"

    def test_classify_communicable(self):
        from src.data.icd10_mapper import ICD10Mapper
        mapper = ICD10Mapper()
        r = mapper.map("loose motion")
        assert r.disease_category == "Communicable"

    def test_classify_symptom_nos(self):
        from src.data.icd10_mapper import ICD10Mapper
        mapper = ICD10Mapper()
        r = mapper.map("fever")
        assert r.disease_category == "Symptom NOS"


# ── CUSUM Tests ────────────────────────────────────────────────────────────────

class TestCUSUM:
    def test_no_alert_on_stable_series(self):
        import pandas as pd
        import numpy as np
        from src.pipeline.trend_detector import CUSUMDetector

        detector = CUSUMDetector()
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        stable = pd.Series(np.ones(30) * 10 + np.random.RandomState(0).randn(30), index=dates)
        scores, alerts = detector.fit_predict(stable)
        assert not alerts.iloc[-1]

    def test_alert_on_spike(self):
        import pandas as pd
        import numpy as np
        from src.pipeline.trend_detector import CUSUMDetector

        detector = CUSUMDetector()
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        # Normal for 20 days, then spike
        values = np.concatenate([np.ones(20) * 5, np.ones(10) * 30])
        series = pd.Series(values, index=dates)
        scores, alerts = detector.fit_predict(series)
        assert alerts.iloc[-1]   # Should alert at end of spike


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
