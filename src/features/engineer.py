import polars as pl
from pathlib import Path
from config import PROCESSED_DIR,TIME_COL,LOCATION_COL,TARGET_COL
from src.utils.logger import logger
import numpy as np

def add_time_features(df:pl.DataFrame)->pl.DataFrame:
    time_df=df.with_columns(
        [
            pl.col("pickup_hour")
            .dt.hour()
            .alias("hour_of_day"),

           pl.col("pickup_hour")
            .dt.weekday()
            .alias("day_of_week"),

            pl.col("pickup_hour")
            .dt.month()
            .alias("month"),

            (pl.col("pickup_hour").dt.weekday()>=5)
            .alias("is_weekend"),

            pl.col("pickup_hour")
            .dt.hour()
            .is_in([7,8,9,17,18,19])
            .alias("is_rush_hour"),

            (
                2 * np.pi *
                pl.col("pickup_hour").dt.hour() / 24
            )
            .sin()
            .alias("hour_sin"),

            (
                2 * np.pi *
                pl.col("pickup_hour").dt.hour() / 24
            )
            .cos()
            .alias("hour_cos"),

            (
                2 * np.pi *
                pl.col("pickup_hour").dt.weekday() / 7
            )
            .sin()
            .alias("dow_sin"),

            (
                2 * np.pi *
                pl.col("pickup_hour").dt.weekday() / 7
            )
            .cos()
            .alias("dow_cos"),
        ]
    )
    return time_df

def add_lag_features(df:pl.DataFrame)->pl.DataFrame:
    lag_df=df.with_columns(
        [
            pl.col("trip_count")
            .shift(1)
            .over("PULocationID")
            .alias("lag_1h"),

            pl.col("trip_count")
            .shift(24)
            .over("PULocationID")
            .alias("lag_24h"),

            pl.col("trip_count")
            .shift(168)
            .over("PULocationID")
            .alias("lag_168h"),
        ]
    )
    return lag_df

def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    def compute_rolling(group: pl.DataFrame) -> pl.DataFrame:
        return group.with_columns([
            pl.col("trip_count")
            .shift(1)
            .rolling_mean(3,min_samples=1)
            .alias("rolling_mean_3h"),
            
            pl.col("trip_count")
            .shift(1)
            .rolling_mean(24,min_samples=1)
            .alias("rolling_mean_24h"),

            pl.col("trip_count")
            .shift(1)
            .rolling_mean(168)
            .over("PULocationID")
            .alias("rolling_mean_168h"),

            pl.col("trip_count")
            .shift(1)
            .rolling_std(window_size=24)
            .over("PULocationID")
            .alias("rolling_std_24h"),

            pl.col("trip_count")
            .shift(1)
            .rolling_std(168)
            .over("PULocationID")
            .alias("rolling_std_168h")
        ])
    
    result = (
        df.sort(["PULocationID", "pickup_hour"])
        .group_by("PULocationID")
        .map_groups(compute_rolling)
        .sort(["PULocationID", "pickup_hour"])
    )
    return result

def drop_nulls(df:pl.DataFrame)->pl.DataFrame:
    df=df.drop_nulls(subset=[
        'lag_1h',
        'lag_24h',
        'lag_168h',
        'rolling_mean_3h',
        'rolling_mean_24h',
        'rolling_mean_168h',
        'rolling_std_24h',
        'rolling_std_168h'
        ])
    return df

def build_features(df:pl.DataFrame)->pl.DataFrame:
    df=df.sort([LOCATION_COL,TIME_COL])
    logger.info("Adding time features")
    df=add_time_features(df)
    logger.info("Successfully added time features")
    logger.info("Adding Lag features")
    df=add_lag_features(df)
    logger.info("Successfully added lag features")
    logger.info("Adding rolling features")
    df=add_rolling_features(df)
    logger.info("Successfully added rolling features")
    df=drop_nulls(df)
    logger.info(f"completed feature engineering new data shape {df.shape[0]} rows & {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df = pl.read_parquet(PROCESSED_DIR / "train.parquet")
    df = build_features(df)
    print(df.shape)
    print(df.head())
    df.write_parquet(PROCESSED_DIR / "train_features.parquet")