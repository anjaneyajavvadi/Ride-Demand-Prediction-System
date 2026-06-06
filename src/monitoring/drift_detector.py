import json
import pandas as pd
import polars as pl
from datetime import datetime
from pathlib import Path
from evidently import Report,DataDefinition,Dataset
from evidently.presets import DataDriftPreset
from config import (
    PROCESSED_DIR,DRIFT_SCORE_THRESHOLD,
    REFERENCE_DIR,DRIFT_COLUMNS,CURRENT_REFERENCE,ORIGINAL_REFERENCE,
    DRIFT_COLUMN_COUNT_THRESHOLD,DRIFT_WINDOW_COUNT_THRESHOLD
)
from src.features.schema import FEATURE_COLUMNS
from src.utils.logger import logger
from src.serving.feature_store import FeatureStore

featurestore=FeatureStore()

ZONE_HOUR_AVG_DF:pd.DataFrame=pd.read_parquet(
    PROCESSED_DIR / "zone_hour_avg.parquet"
)

_drift_counter=0
_calm_counter = 0  
_REFERENCE_DF=None

def check_drift(batch: list[dict]) -> bool:
    global _drift_counter,_calm_counter

    curr_df = _batch_to_dataframe(batch)
    if curr_df is None or len(curr_df) == 0:
        return False

    report = Report(metrics=[DataDriftPreset()])
    run_result = report.run(
        reference_data=get_reference_df(),
        current_data=curr_df
    )

    result = json.loads(run_result.json())
    drift_scores = _extract_drift_scores(result)

    for col, score in drift_scores.items():
        logger.info(f"  {col}: {score:.4f}")

    drifted_columns = {
        col: score 
        for col, score in drift_scores.items() 
        if score > DRIFT_SCORE_THRESHOLD
    }
    drifted_count = len(drifted_columns)

    is_drifted = drifted_count > DRIFT_COLUMN_COUNT_THRESHOLD

    logger.info(
        f"Columns drifted: {drifted_count}/{len(drift_scores)} | "
        f"Drifted columns: {list(drifted_columns.keys())} | "
        f"Trigger: {is_drifted}"
    )

    _save_report(run_result)

    if is_drifted:
        _drift_counter += 1
        _calm_counter = 0
        logger.warning(
            f"Drift detected. Counter: {_drift_counter}/{DRIFT_WINDOW_COUNT_THRESHOLD}"
        )
    else:
        _calm_counter += 1
        if _calm_counter >= 2:  # 2 consecutive calm weeks resets drift counter
            _drift_counter = 0
            logger.info(f"2 consecutive calm weeks. Drift counter reset.")
        else:
            logger.info(f"Calm week {_calm_counter}/2. Drift counter holding at {_drift_counter}.")

    if _drift_counter >= DRIFT_WINDOW_COUNT_THRESHOLD:
        logger.warning("Retraining threshold reached. Triggering pipeline.")
        _drift_counter = 0
        _calm_counter
        return True

    return False

def _batch_to_dataframe(batch: list[dict]) -> pd.DataFrame:
    parsed = []
    for row in batch:
        parsed.append({
            "pickup_hour": datetime.fromisoformat(row["pickup_hour"]),
            "trip_count": row["trip_count"],
            "zone_id": row["zone_id"],
        })

    df = pl.DataFrame(parsed, schema={
        "pickup_hour": pl.Datetime,
        "trip_count": pl.Int64,
        "zone_id": pl.Int64,
    })


    hourly = (
        df.group_by("pickup_hour")
        .agg([
            pl.col("trip_count").sum().alias("total_demand"),
            pl.col("trip_count").mean().alias("avg_demand"),
            pl.col("trip_count").std().alias("std_demand"),
            pl.col("trip_count").max().alias("peak_demand"),
        ])
        .sort("pickup_hour")
    )


    return hourly.select(DRIFT_COLUMNS).to_pandas()


def _extract_drift_share(result: dict) -> float:
    try:
        return float(result["metrics"][0]["value"]["share"])
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Failed to extract drift share: {e}")
        return 0.0
    
def _extract_drift_scores(result: dict) -> dict:
    """Extract per-column drift scores."""
    scores = {}
    try:
        for metric in result["metrics"][1:]:  # skip index 0 (DriftedColumnsCount)
            name = metric["metric_name"]
            value = metric["value"]
            # extract column name from metric_name string
            # format: "ValueDrift(column=total_demand,method=...,threshold=...)"
            if "column=" in name:
                col = name.split("column=")[1].split(",")[0]
                scores[col] = float(value)
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Failed to extract drift scores: {e}")
    return scores

def _save_report(report) -> None:
    reports_dir = Path("artifacts/drift_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT_%H%M%S")
    path = reports_dir / f"drift_report_{timestamp}.html"
    try:
        report.save_html(str(path))
    except AttributeError:
    
        with open(path, "w") as f:
            f.write(report._inner_suite.get_html())
    logger.info(f"Drift report saved: {path}")


def get_reference_df() -> pd.DataFrame:
    global _REFERENCE_DF
    if _REFERENCE_DF is None:
        # use current if exists, fall back to original
        ref_path = CURRENT_REFERENCE if CURRENT_REFERENCE.exists() else ORIGINAL_REFERENCE
        _REFERENCE_DF = pd.read_parquet(ref_path)[DRIFT_COLUMNS]
        logger.info(f"Loaded reference from {ref_path.name}")
    return _REFERENCE_DF