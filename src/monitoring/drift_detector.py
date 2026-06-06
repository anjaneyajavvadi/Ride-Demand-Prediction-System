import json
import pandas as pd
import polars as pl
from datetime import datetime
from pathlib import Path
from evidently import Report,DataDefinition,Dataset
from evidently.presets import DataDriftPreset
from config import (
    PROCESSED_DIR,DRIFT_SHARE_THRESHOLD,
    DRIFT_WINDOW_COUNT_THRESHOLD,REFERENCE_DIR,DRIFT_COLUMNS
)
from src.features.schema import FEATURE_COLUMNS
from src.utils.logger import logger
from src.serving.feature_store import FeatureStore

featurestore=FeatureStore()

ZONE_HOUR_AVG_DF:pd.DataFrame=pd.read_parquet(
    PROCESSED_DIR / "zone_hour_avg.parquet"
)

REFERENCE_DF:pd.DataFrame=pd.read_parquet(
    REFERENCE_DIR / "reference.parquet"
)[DRIFT_COLUMNS]

_drift_counter=0

def check_drift(batch:list[dict])->bool:

    global _drift_counter

    logger.info("drift detction")

    curr_df=_batch_to_dataframe(batch)
    if curr_df is None or len(curr_df)==0:
        return False
    
    report=Report(
        metrics=[DataDriftPreset()]
    )

    run_result=report.run(
        reference_data=REFERENCE_DF,
        current_data=curr_df
    )

    result = json.loads(run_result.json())

    drift_share=_extract_drift_share(result)
    is_drifted=drift_share>DRIFT_SHARE_THRESHOLD

    logger.info(f"Drift share: {drift_share:.3f} | Drifted: {is_drifted}")

    _save_report(run_result)

    if is_drifted:
        _drift_counter += 1
        logger.warning(f"Drift detected. Counter: {_drift_counter}/{DRIFT_WINDOW_COUNT_THRESHOLD}")
    else:
        _drift_counter = 0

    if _drift_counter >= DRIFT_WINDOW_COUNT_THRESHOLD:
        logger.warning("Retraining threshold reached. Triggering pipeline.")
        _drift_counter = 0
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


        


    
