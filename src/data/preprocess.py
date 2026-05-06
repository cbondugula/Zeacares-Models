"""
ZeaCares Data Preprocessing & PII Anonymization
Cleans raw clinical text, strips PII, extracts structured fields.
"""
import re
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Age-band boundaries (10-year bands for DPDP Act compliance)
AGE_BANDS = [(0, 14), (15, 24), (25, 34), (35, 44), (45, 54),
             (55, 64), (65, 74), (75, 84), (85, 120)]

def _age_to_band(age: int) -> str:
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "Unknown"


@dataclass
class ClinicalRecord:
    raw_text: str
    clean_text: str
    gender: Optional[str]
    age_band: Optional[str]
    disease_raw: Optional[str]
    onset: Optional[str]
    duration_days: Optional[int]
    severity: Optional[str]
    temp_f: Optional[float]
    pulse: Optional[int]
    bp_sys: Optional[int]
    bp_dia: Optional[int]
    spo2: Optional[float]
    bmi_status: Optional[str]
    facility: Optional[str]
    district: Optional[str]
    patient_type: Optional[str]


# Regex patterns for structured fields
_PATTERNS = {
    "gender_age": re.compile(r"(Male|Female)\s+(\d+)\s+years?", re.IGNORECASE),
    "disease":    re.compile(r"presented with\s+(.+?)\.\s+Onset", re.IGNORECASE),
    "onset":      re.compile(r"Onset was\s+(\w+)", re.IGNORECASE),
    "duration":   re.compile(r"duration of\s+(\d+)\s+day", re.IGNORECASE),
    "severity":   re.compile(r"Severity:\s+(\w+)", re.IGNORECASE),
    "temp":       re.compile(r"Temperature\s+([\d.]+)F", re.IGNORECASE),
    "pulse":      re.compile(r"Pulse\s+(\d+)bpm", re.IGNORECASE),
    "bp":         re.compile(r"BP\s+(\d+)/(\d+)mmHg", re.IGNORECASE),
    "spo2":       re.compile(r"SpO2\s+([\d.]+)%", re.IGNORECASE),
    "bmi":        re.compile(r"BMI status:\s+(\w+)", re.IGNORECASE),
    "facility":   re.compile(r"Attended\s+(.+?)\.?\s*$", re.IGNORECASE),
}


def extract_fields(text: str) -> dict:
    fields = {k: None for k in ["gender", "age", "disease_raw", "onset",
                                  "duration_days", "severity", "temp_f",
                                  "pulse", "bp_sys", "bp_dia", "spo2",
                                  "bmi_status", "facility"]}

    m = _PATTERNS["gender_age"].search(text)
    if m:
        fields["gender"] = m.group(1)[0].upper()  # "M" or "F"
        fields["age"] = int(m.group(2))

    m = _PATTERNS["disease"].search(text)
    if m:
        fields["disease_raw"] = m.group(1).strip().lower()

    m = _PATTERNS["onset"].search(text)
    if m:
        fields["onset"] = m.group(1).lower()

    m = _PATTERNS["duration"].search(text)
    if m:
        fields["duration_days"] = int(m.group(1))

    m = _PATTERNS["severity"].search(text)
    if m:
        fields["severity"] = m.group(1).lower()

    m = _PATTERNS["temp"].search(text)
    if m:
        fields["temp_f"] = float(m.group(1))

    m = _PATTERNS["pulse"].search(text)
    if m:
        fields["pulse"] = int(m.group(1))

    m = _PATTERNS["bp"].search(text)
    if m:
        fields["bp_sys"] = int(m.group(1))
        fields["bp_dia"] = int(m.group(2))

    m = _PATTERNS["spo2"].search(text)
    if m:
        fields["spo2"] = float(m.group(1))

    m = _PATTERNS["bmi"].search(text)
    if m:
        fields["bmi_status"] = m.group(1).capitalize()

    m = _PATTERNS["facility"].search(text)
    if m:
        fields["facility"] = m.group(1).strip()

    return fields


def anonymize_text(text: str, gender: Optional[str], age: Optional[int]) -> str:
    """Replace exact age+gender with anonymized age band."""
    clean = text
    if gender and age:
        age_band = _age_to_band(age)
        gender_full = "Male" if gender == "M" else "Female"
        clean = re.sub(
            rf"{gender_full}\s+{age}\s+years?",
            f"[{gender}, {age_band}]",
            clean,
            flags=re.IGNORECASE,
        )
    # Strip facility name (location PII)
    clean = re.sub(r"Attended\s+\S+.*$", "Attended [FACILITY]", clean, flags=re.IGNORECASE)
    return clean


def process_record(row: pd.Series) -> ClinicalRecord:
    text = str(row.get("clinicalText", ""))
    district = str(row.get("district", "Unknown"))
    patient_type = str(row.get("patientType", "Unknown"))

    fields = extract_fields(text)
    age = fields.pop("age", None)
    age_band = _age_to_band(age) if age is not None else None
    clean_text = anonymize_text(text, fields.get("gender"), age)

    return ClinicalRecord(
        raw_text=text,
        clean_text=clean_text,
        gender=fields["gender"],
        age_band=age_band,
        disease_raw=fields["disease_raw"],
        onset=fields["onset"],
        duration_days=fields["duration_days"],
        severity=fields["severity"],
        temp_f=fields["temp_f"],
        pulse=fields["pulse"],
        bp_sys=fields["bp_sys"],
        bp_dia=fields["bp_dia"],
        spo2=fields["spo2"],
        bmi_status=fields["bmi_status"],
        facility=fields["facility"],
        district=district,
        patient_type=patient_type,
    )


def preprocess_dataset(input_path: str, output_path: Optional[str] = None) -> list[dict]:
    df = pd.read_excel(input_path) if input_path.endswith(".xlsx") else pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records from {input_path}")

    records = []
    for _, row in df.iterrows():
        try:
            rec = process_record(row)
            records.append(asdict(rec))
        except Exception as e:
            logger.warning(f"Failed to process row: {e}")

    logger.info(f"Processed {len(records)} records successfully")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info(f"Saved to {output_path}")

    return records


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/zeacares_upload_ready.csv.xlsx"
    records = preprocess_dataset(input_file, "results/preprocessed.json")

    # Quick stats
    df = pd.DataFrame(records)
    print(f"\n=== Preprocessing Stats ===")
    print(f"Total records:      {len(df)}")
    print(f"Gender extracted:   {df['gender'].notna().sum()} ({df['gender'].notna().mean()*100:.1f}%)")
    print(f"Disease extracted:  {df['disease_raw'].notna().sum()} ({df['disease_raw'].notna().mean()*100:.1f}%)")
    print(f"BP extracted:       {df['bp_sys'].notna().sum()} ({df['bp_sys'].notna().mean()*100:.1f}%)")
    print(f"Severity extracted: {df['severity'].notna().sum()} ({df['severity'].notna().mean()*100:.1f}%)")
    print(f"\nTop 10 diseases:")
    print(df['disease_raw'].value_counts().head(10).to_string())
