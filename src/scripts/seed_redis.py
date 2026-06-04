import polars as pl
from src.serving.feature_store import FeatureStore
from config import PROCESSED_DIR

df=pl.read_parquet(
    PROCESSED_DIR/"stream.parquet"
)
feature_store=FeatureStore()
for i,row in enumerate(df.iter_rows(named=True)):
    feature_store.set_trip_count(
        zone_id=row['PULocationID'],
        timestamp=row['pickup_hour'],
        count=row['trip_count']
    )
    if i%10000==0:
        print(f"{i}th row seeding completed")