import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
import mlflow
from config import VALIDATION_SPLIT_DATE,TARGET_COL,MLFLOW_TRACKING_URI,MLFLOW_EXPERIMENT,MODEL_NAME,XGB_PARAMS,PROCESSED_DIR,ARTIFACTS_DIR,RANDOM_SEED,REFERENCE_DIR,DRIFT_COLUMNS
from src.training.evaluate import compute_metrics,plot_predictions
from src.features.schema import FEATURE_COLUMNS,FEATURE_SCHEMA
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
from src.utils.logger import logger

def load_training_data(df=pl.DataFrame)->tuple:

    logger.info("Creating train and validation split")
    train_df=df.filter(pl.col("pickup_hour")<VALIDATION_SPLIT_DATE)
    val_df=df.filter(pl.col("pickup_hour")>=VALIDATION_SPLIT_DATE)

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

    reference=train_df.select(DRIFT_COLUMNS).sample(n=5000,seed=RANDOM_SEED)
    reference.write_parquet(REFERENCE_DIR / "reference.parquet")
        
    zone_hour_avg.write_parquet(PROCESSED_DIR / "zone_hour_avg.parquet")
    
    X_train=train_df.select(FEATURE_COLUMNS).to_pandas()
    y_train=train_df[TARGET_COL].to_pandas()
    X_val=val_df.select(FEATURE_COLUMNS).to_pandas()
    y_val=val_df[TARGET_COL].to_pandas()

    X_train['PULocationID']=X_train['PULocationID'].astype("category")
    X_val["PULocationID"]=X_val["PULocationID"].astype("category")
    logger.info("Created train and validation splits")

    return X_train,y_train,X_val,y_val,val_df

def train(df):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train,y_train,X_val,y_val,val_df=load_training_data(df)


    with mlflow.start_run():
        mlflow.set_tags({
            "model":"XGBoost",
            "version":"0.1",
            "name":"ride_analysis_xgboost"
        })

        mlflow.log_params(
            XGB_PARAMS
        )

        mlflow.log_params({
            "dataset_rows": df.shape[0],
            "dataset_columns": df.shape[1],
            "feature_count": len(FEATURE_COLUMNS),
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
        })

        mlflow.log_dict(
            {"features": FEATURE_COLUMNS},
            "feature_columns.json"
        )
        mlflow.log_artifact(str(PROCESSED_DIR / "zone_hour_avg.parquet"))
        mlflow.log_artifact(str(REFERENCE_DIR / "reference.parquet"))

        logger.info("Initialising XGBoost model")

        model=xgb.XGBRegressor(
            **XGB_PARAMS,
            enable_categorical=True,
            tree_method='hist',
            early_stopping_rounds=50
        )

        

        logger.info("Model training started")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )


        logger.info("Model training Successful, computing metrics")
        train_pred = model.predict(X_train)
        train_metrics = compute_metrics(y_train, train_pred)

        val_pred = model.predict(X_val)
        val_metrics = compute_metrics(y_val, val_pred)

        run_id = mlflow.active_run().info.run_id
        
        save_path=plot_predictions(val_df,val_pred,zone_id=202,save_path=ARTIFACTS_DIR/"plots"/f"val_forecast_{run_id}.png",title="validation actual vs prediction plot")
        train_metrics = {
            f"train_{k}": v
            for k, v in train_metrics.items()
        }

        val_metrics = {
            f"val_{k}": v
            for k, v in val_metrics.items()
        }
        lag1_rmse=root_mean_squared_error(
            y_val,
            X_val["lag_1h"]
        )

        lag24_rmse=root_mean_squared_error(
                y_val,
                X_val["lag_24h"]
            )
        lag168_rmse=root_mean_squared_error(
                y_val,
                X_val["lag_168h"]
            )
        
        baseline_metrics = {
        "baseline_lag1_rmse": lag1_rmse,
        "baseline_lag24_rmse": lag24_rmse,
            "baseline_lag168_rmse": lag168_rmse,
        }

        mlflow.log_metrics(baseline_metrics)
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(val_metrics)

        mlflow.log_artifact(str(save_path))

        mlflow.xgboost.log_model(
            model,
            name="model",
                registered_model_name="ride_demand_xgboost"
        )


if __name__=="__main__":
    df=pl.read_parquet(PROCESSED_DIR/"train_features.parquet")
    train(df)


