import polars as pl
from pathlib import Path
from config import (
    RAW_DIR,PROCESSED_DIR,YEAR,TAXI_TYPE,
    COLUMNS_TO_KEEP,TRAIN_MONTHS,STREAM_MONTHS,
    TIME_COL,LOCATION_COL,TARGET_COL
    )
from src.utils.logger import logger
import datetime

def load_and_clean(path:Path)->pl.DataFrame:
    df=pl.read_parquet(path,columns=COLUMNS_TO_KEEP)
    df=df.rename({"tpep_pickup_datetime":"pickup_dt","tpep_dropoff_datetime":"dropoff_dt"})
    df=df.drop_nulls(subset=['pickup_dt'])
    df=df.filter(
        (pl.col("trip_distance")>0) &
        (pl.col("fare_amount")>0 ) &
        (pl.col("PULocationID").is_between(1,263)) &
        (pl.col("pickup_dt").dt.year()==2025)
    )
    return df

def aggregate_to_hourly(df:pl.DataFrame)->pl.DataFrame:

    agg_df=(
        df.with_columns(
            pl.col("pickup_dt")
            .dt.truncate("1h")
            .alias("pickup_hour")
        )
        .group_by(["pickup_hour","PULocationID"])
        .len()
        .rename({"len":"trip_count"})
    )

    all_hours = pl.datetime_range(
        start=df["pickup_dt"].min().replace(minute=0, second=0, microsecond=0),
        end=df["pickup_dt"].max().replace(minute=0, second=0, microsecond=0),
        interval="1h",
        eager=True
    ).alias("pickup_hour")

    all_zones = pl.Series("PULocationID", list(range(1, 264)))

    full_grid = all_hours.to_frame().join(all_zones.to_frame(), how="cross")

    result = (
        full_grid
        .join(agg_df, on=["pickup_hour", "PULocationID"], how="left")
        .with_columns(pl.col("trip_count").fill_null(0))
        .sort(["pickup_hour", "PULocationID"])
    )

    return result

def process_all_months(months:list[int])->pl.DataFrame:
    frames=[]
    for month in months:
        path=RAW_DIR/f"{TAXI_TYPE}_tripdata_{YEAR}-{month:02d}.parquet"
        logger.info(f"Processing {path.name}")
        df=load_and_clean(path)
        df=aggregate_to_hourly(df)
        frames.append(df)
        logger.info(f"Month {month}: {len(df)} hourly records")

    return pl.concat(frames).sort([TIME_COL,LOCATION_COL])

def split_train_stream(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    split_date = datetime.datetime(2025, 7, 1)

    train_df=df.filter(
        pl.col("pickup_hour")<split_date
    )
    stream_df=df.filter(
        pl.col("pickup_hour")>=split_date
    )
    return train_df,stream_df

def save_splits(train_df:pl.DataFrame,stream_df:pl.DataFrame)->None:
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    train_df.write_parquet(PROCESSED_DIR / "train.parquet")
    stream_df.write_parquet(PROCESSED_DIR / "stream.parquet")
    logger.info(f"Train: {len(train_df)} rows | Stream: {len(stream_df)} rows")


if __name__ == "__main__":
    logger.info("Processing all months...")
    data = process_all_months(TRAIN_MONTHS+STREAM_MONTHS)

    train_df,stream_df=split_train_stream(data)

    logger.info("Saving train and stream data")
    save_splits(train_df,stream_df)

        