"""
Trend Detection Engine
Detects disease outbreaks using CUSUM + Facebook Prophet.
Runs on classified records; generates alerts when thresholds are exceeded.
"""
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import numpy as np

logger = logging.getLogger(__name__)

CUSUM_THRESHOLD = 5.0       # Alert when CUSUM score exceeds this
CUSUM_K = 0.5               # CUSUM slack parameter (sensitivity)
MIN_CASES_FOR_ALERT = 5     # Minimum absolute cases to trigger alert
MIN_DAYS_DATA = 7           # Minimum days of data needed to run detection
PROPHET_CONFIDENCE = 0.95   # Forecast interval confidence


@dataclass
class TrendAlert:
    district: str
    disease_category: str
    alert_type: str              # "cusum_alert" | "prophet_anomaly" | "combined"
    alert_severity: str          # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    current_cases_7d: int
    expected_cases_7d: float
    cusum_score: float
    prophet_anomaly: bool
    percent_above_baseline: float
    triggered_at: str
    details: str


@dataclass
class TrendSummary:
    district: str
    disease_category: str
    date_range: str
    total_cases: int
    daily_avg: float
    peak_day: Optional[str]
    peak_count: int
    cusum_score: float
    alerts: list[TrendAlert]


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) control chart for outbreak detection.
    Standard epidemiological method used by CDC and WHO.
    Detects sustained upward shifts in disease incidence.
    """

    def __init__(self, k: float = CUSUM_K, threshold: float = CUSUM_THRESHOLD):
        self.k = k           # Allowable slack (slack = k * sigma)
        self.threshold = threshold

    def fit_predict(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Returns (cusum_scores, alerts) for each time point.
        cusum_score > threshold → outbreak suspected.
        """
        if len(series) < MIN_DAYS_DATA:
            return pd.Series(np.zeros(len(series))), pd.Series(np.zeros(len(series), dtype=bool))

        mean = series[:max(7, len(series)//3)].mean()  # baseline from first third
        std = series[:max(7, len(series)//3)].std() + 1e-8
        k_abs = self.k * std

        cusum = 0.0
        scores = []
        alerts = []
        for val in series:
            cusum = max(0.0, cusum + (val - mean - k_abs))
            scores.append(cusum)
            alerts.append(cusum > self.threshold)

        return pd.Series(scores, index=series.index), pd.Series(alerts, index=series.index)

    def get_current_score(self, series: pd.Series) -> float:
        scores, _ = self.fit_predict(series)
        return float(scores.iloc[-1]) if len(scores) > 0 else 0.0


class ProphetDetector:
    """
    Facebook Prophet for time-series anomaly detection.
    Forecasts expected case counts and flags deviations as anomalies.
    """

    def __init__(self, interval_width: float = PROPHET_CONFIDENCE):
        self.interval_width = interval_width
        self._prophet_available = False
        try:
            from prophet import Prophet  # noqa
            self._prophet_available = True
        except ImportError:
            logger.warning("Prophet not installed — using statistical fallback for anomaly detection")

    def detect_anomalies(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Returns (forecast_upper, is_anomaly) series.
        Points above forecast upper bound are anomalies.
        """
        if len(series) < MIN_DAYS_DATA:
            return pd.Series(series * 1.5), pd.Series(np.zeros(len(series), dtype=bool))

        if not self._prophet_available:
            return self._statistical_fallback(series)

        try:
            from prophet import Prophet
            df_prophet = pd.DataFrame({
                "ds": series.index,
                "y": series.values,
            })
            model = Prophet(
                interval_width=self.interval_width,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=len(series) > 60,
                changepoint_prior_scale=0.05,
            )
            model.fit(df_prophet)
            forecast = model.predict(df_prophet)
            forecast.index = series.index

            upper = forecast["yhat_upper"]
            is_anomaly = series > upper
            return upper, is_anomaly
        except Exception as e:
            logger.warning(f"Prophet failed: {e} — using statistical fallback")
            return self._statistical_fallback(series)

    def _statistical_fallback(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Rolling mean + 2 std as baseline when Prophet unavailable."""
        window = min(7, len(series) - 1)
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std().fillna(1.0)
        upper = rolling_mean + 2 * rolling_std
        is_anomaly = series > upper
        return upper, is_anomaly


class TrendDetector:
    """
    Main trend detection orchestrator.
    Aggregates classified records by district + disease category,
    then runs CUSUM + Prophet to generate outbreak alerts.
    """

    def __init__(self):
        self.cusum = CUSUMDetector()
        self.prophet = ProphetDetector()

    def load_classified_records(self, path: str) -> pd.DataFrame:
        """Load classified records from classifier.py output."""
        with open(path) as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        if "date" not in df.columns:
            # Simulate dates for demo (in production, date comes from actual records)
            np.random.seed(42)
            base = datetime(2026, 2, 1)
            df["date"] = [
                (base + timedelta(days=int(np.random.randint(0, 90)))).strftime("%Y-%m-%d")
                for _ in range(len(df))
            ]
        df["date"] = pd.to_datetime(df["date"])
        return df

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate case counts by date + district + disease_category."""
        agg = (
            df.groupby(["date", "district", "disease_category"])
            .size()
            .reset_index(name="case_count")
        )
        return agg

    def detect_for_district_disease(self, series: pd.Series,
                                     district: str,
                                     disease_category: str) -> Optional[TrendAlert]:
        """Run CUSUM + Prophet on a single district-disease time series."""
        if series.empty or series.sum() < MIN_CASES_FOR_ALERT:
            return None

        cusum_score = self.cusum.get_current_score(series)
        forecast_upper, prophet_anomaly = self.prophet.detect_anomalies(series)
        is_prophet_anomaly = bool(prophet_anomaly.iloc[-1]) if len(prophet_anomaly) > 0 else False

        cusum_alert = cusum_score > CUSUM_THRESHOLD
        if not (cusum_alert or is_prophet_anomaly):
            return None

        current_7d = int(series.tail(7).sum())
        baseline_7d = float(series.head(min(14, len(series)//2)).mean() * 7)
        pct_above = (current_7d - baseline_7d) / max(baseline_7d, 1) * 100

        if cusum_score > CUSUM_THRESHOLD * 2 or pct_above > 100:
            severity = "CRITICAL"
        elif cusum_score > CUSUM_THRESHOLD * 1.5 or pct_above > 50:
            severity = "HIGH"
        elif cusum_score > CUSUM_THRESHOLD or is_prophet_anomaly:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        alert_type = (
            "combined" if cusum_alert and is_prophet_anomaly
            else "cusum_alert" if cusum_alert
            else "prophet_anomaly"
        )

        return TrendAlert(
            district=district,
            disease_category=disease_category,
            alert_type=alert_type,
            alert_severity=severity,
            current_cases_7d=current_7d,
            expected_cases_7d=round(baseline_7d, 1),
            cusum_score=round(cusum_score, 2),
            prophet_anomaly=is_prophet_anomaly,
            percent_above_baseline=round(pct_above, 1),
            triggered_at=datetime.utcnow().isoformat() + "Z",
            details=f"{disease_category} cases in {district} are {pct_above:.0f}% above baseline",
        )

    def run(self, classified_records_path: str,
            output_path: str = "results/alerts.json",
            district_filter: Optional[str] = None) -> list[TrendAlert]:
        df = self.load_classified_records(classified_records_path)
        if district_filter:
            df = df[df["district"] == district_filter]

        agg = self.aggregate_daily(df)
        all_alerts = []
        summaries = []

        for (district, category), group in agg.groupby(["district", "disease_category"]):
            # Build complete date range time series (fill missing days with 0)
            full_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
            series = group.set_index("date")["case_count"].reindex(full_range, fill_value=0)

            alert = self.detect_for_district_disease(series, str(district), str(category))
            if alert:
                all_alerts.append(alert)

            # Build summary regardless of alert
            summaries.append(TrendSummary(
                district=str(district),
                disease_category=str(category),
                date_range=f"{group['date'].min().date()} to {group['date'].max().date()}",
                total_cases=int(series.sum()),
                daily_avg=round(float(series.mean()), 1),
                peak_day=str(series.idxmax().date()) if len(series) > 0 else None,
                peak_count=int(series.max()),
                cusum_score=round(self.cusum.get_current_score(series), 2),
                alerts=[a for a in all_alerts if a.district == district and a.disease_category == category],
            ))

        # Sort alerts by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        all_alerts.sort(key=lambda a: (severity_order.get(a.alert_severity, 4),
                                       -a.percent_above_baseline))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "total_alerts": len(all_alerts),
                "alerts": [asdict(a) for a in all_alerts],
                "summaries": [asdict(s) for s in summaries],
            }, f, indent=2, default=str)

        self._print_alerts(all_alerts)
        logger.info(f"Saved {len(all_alerts)} alerts to {output_path}")
        return all_alerts

    def _print_alerts(self, alerts: list[TrendAlert]) -> None:
        if not alerts:
            print("\n✅ No outbreak alerts detected.")
            return
        print(f"\n⚠️  {len(alerts)} Outbreak Alert(s) Detected:")
        print("-" * 70)
        for a in alerts:
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(a.alert_severity, "⚪")
            print(f"{icon} [{a.alert_severity}] {a.district} — {a.disease_category}")
            print(f"   Cases (7d): {a.current_cases_7d} vs expected {a.expected_cases_7d:.0f} "
                  f"(+{a.percent_above_baseline:.0f}%)")
            print(f"   CUSUM: {a.cusum_score:.1f} | Prophet anomaly: {a.prophet_anomaly}")
            print()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/classified.json")
    parser.add_argument("--output", default="results/alerts.json")
    parser.add_argument("--district", type=str, help="Filter to specific district")
    args = parser.parse_args()

    detector = TrendDetector()
    alerts = detector.run(args.input, args.output, args.district)
    print(f"\nTotal alerts generated: {len(alerts)}")
