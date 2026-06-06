import mlflow
import mlflow.xgboost
import xgboost as xgb
import polars as pl
import requests
from datetime import datetime,timedelta
from config import (
    PROCESSED_DIR,MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,XGB_PARAMS,
    TARGET_COL,RMSE_IMPROVEMENT_THRESHOLD,
    API_HOST,API_PORT,MODEL_NAME,REFERENCE_DIR,CURRENT_REFERENCE
)
from src.features.engineer import build_features
from src.features.schema import FEATURE_COLUMNS
from src.training.evaluate import compute_metrics
from src.utils.logger import logger

def load_retrain_data(trigger_timestamp:datetime)->pl.DataFrame:
    train_df=pl.read_parquet(PROCESSED_DIR/"train.parquet")
    stream_df=pl.read_parquet(PROCESSED_DIR/"stream.parquet")

    stream_df=stream_df.filter(
        pl.col("pickup_hour")<=trigger_timestamp
    )

    df=pl.concat([train_df,stream_df]).sort(["PULocationID","pickup_hour"])
    logger.info(f"Retrain data: {len(df)} rows up to {trigger_timestamp}")
    return df

def get_current_model_rmse() -> float:
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME)
        if not versions:
            return float("inf")
        run_id = versions[-1].run_id
        run = client.get_run(run_id)
        return float(run.data.metrics["val_rmse"])
    except Exception as e:
        logger.warning(f"Could not get current RMSE: {e}. Defaulting to inf.")
        return float("inf")
    
def retrain_pipeline(trigger_timestamp:datetime=None):
    if trigger_timestamp is None:
        trigger_timestamp=datetime.now()
    
    logger.info(f"Retraining triggered at {trigger_timestamp}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df=load_retrain_data(trigger_timestamp)
    df=build_features(df)

    cutoff=trigger_timestamp-timedelta(days=30)
    train_df=df.filter(pl.col("pickup_hour")<cutoff)
    val_df=df.filter(pl.col("pickup_hour")>=cutoff)

    zone_hour_avg = (
        train_df
        .group_by(
            ["PULocationID", "hour_of_day"]
        )
        .agg(
            pl.mean("trip_count")
            .alias("zone_hour_avg")
        )
    )

    train_df = train_df.join(
        zone_hour_avg,
        on=["PULocationID", "hour_of_day"],
        how="left"
    )
    val_df = val_df.join(
        zone_hour_avg,
        on=["PULocationID", "hour_of_day"],
        how="left"
    )

    logger.info("Updating drift reference dataset")
    hourly_ref = (
        train_df.group_by("pickup_hour")
        .agg([
            pl.col("trip_count").sum().alias("total_demand"),
            pl.col("trip_count").mean().alias("avg_demand"),
            pl.col("trip_count").std().alias("std_demand"),
            pl.col("trip_count").max().alias("peak_demand"),
        ])
        .sort("pickup_hour")
        .drop("pickup_hour")
    )
    hourly_ref.write_parquet(CURRENT_REFERENCE)
    logger.info("Drift reference updated")


    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    X_train = train_df.select(FEATURE_COLUMNS).to_pandas()
    y_train = train_df[TARGET_COL].to_pandas()
    X_val   = val_df.select(FEATURE_COLUMNS).to_pandas()
    y_val   = val_df[TARGET_COL].to_pandas()

    X_train['PULocationID']=X_train['PULocationID'].astype("category")
    X_val['PULocationID']=X_val['PULocationID'].astype("category")

    with mlflow.start_run(run_name=f"retrain_{trigger_timestamp.strftime("%Y%m%d_%H%M")}"):
        mlflow.set_tags(
            {
                "trigger":"drift_detection",
                "trigger_timestamp":str(trigger_timestamp)
            }
        )

        mlflow.log_params(XGB_PARAMS)

        mlflow.log_params({
            "dataset_rows": df.shape[0],
            "dataset_columns": df.shape[1],
            "feature_count": len(FEATURE_COLUMNS),
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
        })

        new_model = xgb.XGBRegressor(
            **XGB_PARAMS,
            enable_categorical=True,
            tree_method="hist",
            early_stopping_rounds=50,
        )

        new_model.fit(
            X_train,y_train,
            eval_set=[(X_val,y_val)],
            verbose=100,
        )

        logger.info("Model training Successful, computing metrics")
        train_pred = new_model.predict(X_train)
        train_metrics = compute_metrics(y_train, train_pred)

        val_pred = new_model.predict(X_val)
        val_metrics = compute_metrics(y_val, val_pred)

        train_metrics = {
            f"train_{k}": v
            for k, v in train_metrics.items()
        }

        val_metrics = {
            f"val_{k}": v
            for k, v in val_metrics.items()
        }

        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(val_metrics)

        new_rmse=val_metrics['val_rmse']

        logger.info(f"New model RMSE: {new_rmse:.4f}")
        
        current_rmse = get_current_model_rmse()
        logger.info(f"Current model RMSE: {current_rmse:.4f}")

        if new_rmse<current_rmse*RMSE_IMPROVEMENT_THRESHOLD:
            logger.info("New model is better. Promoting")

            mlflow.xgboost.log_model(
                new_model,
                name="model",
                registered_model_name=MODEL_NAME
            )
            mlflow.log_metric("promoted", 1)

            try:
                response = requests.post(
                    f"http://localhost:{API_PORT}/reload-model"
                )
                logger.info(f"FastAPI reloaded: {response.json()}")
            except Exception as e:
                logger.error(f"FastAPI reload failed: {e}")
        
        else:
            logger.warning(
                f"New model RMSE {new_rmse:.4f} not better than "
                f"current {current_rmse:.4f}. Discarding."
            )
            mlflow.log_metric("promoted", 0)

    return new_rmse

        

