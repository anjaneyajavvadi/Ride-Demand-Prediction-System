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
)[FEATURE_COLUMNS]

_drift_counter=0

def check_drift(batch:list[dict])->bool:

    global _drift_counter

    curr_df=_batch_to_dataframe(batch)

    if curr_df is None or len(curr_df)==0:
        return False
    
    report=Report(
        metrics=[DataDriftPreset()]
    )

    report.run(
        reference_data=REFERENCE_DF,
        current_data=curr_df
    )

    result=report.as_dict()


    logger.info(
        json.dumps(
            result,
            indent=2,
            default=str
        )
    )

    drift_share=_extract_drift_share(result)
    is_drifted=drift_share>DRIFT_SHARE_THRESHOLD

    logger.info(f"Drift share: {drift_share:.3f} | Drifted: {is_drifted}")

    _save_report(report)

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
    df = pl.DataFrame(batch)

    df = df.with_columns([
        pl.col("pickup_hour")
        .str.to_datetime()
        .alias("pickup_hour")
    ])

    df = df.with_columns([
        pl.col("pickup_hour")
        .dt.hour()
        .alias("hour_of_day"),

        pl.col("pickup_hour")
        .dt.weekday()
        .alias("day_of_week"),

        pl.col("pickup_hour")
        .dt.month()
        .alias("month"),

        (
            pl.col("pickup_hour")
            .dt.weekday() >= 5
        ).alias("is_weekend"),

        pl.col("pickup_hour")
        .dt.hour()
        .is_in([7, 8, 9, 17, 18, 19])
        .alias("is_rush_hour"),
    ])

    return df.select(DRIFT_COLUMNS).to_pandas()



def _extract_drift_share(result:dict)->float:
    try:
        return (
            result['metrics'][0]
            ['result']['share_of_drifted_columns']
        )
    except Exception:
        logger.exception(
            "Failed to extract drift share"
        )
        return 0.0


def _save_report(report:Report)->None:

    reports_dir=Path("artifacts/drift_reports")
    reports_dir.mkdir(parents=True,exist_ok=True)
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    path=reports_dir/f"drift_report_{timestamp}.html"
    report.save_html(str(path))
    logger.info(f"Drift report saved: {path}")


        


    
