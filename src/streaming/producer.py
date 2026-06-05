import json
import time
from pathlib import Path
from kafka import KafkaProducer
from config import (
    KAFKA_BOOTSTRAP, KAFKA_TOPIC,
    PROCESSED_DIR, STREAM_REPLAY_SPEED
)
from src.utils.logger import logger
import polars as pl


def get_producer()->KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v:json.dumps(v).encode("utf-8"),
        acks='all',
        retries=3
    )

def serialize_row(row:dict):
    return {
        "zone_id":row['PULocationID'],
        "pickup_hour":row['pickup_hour'].isoformat(),
        "trip_count":row['trip_count'],
    }

def replay_stream(parquet_path:Path=PROCESSED_DIR/'stream.parquet'):
    producer=get_producer()

    df=pl.read_parquet(parquet_path).sort(['pickup_hour',"PULocationID"])

    logger.info(f"Replaying {len(df)} rows to topic '{KAFKA_TOPIC}'")

    prev_timestamp=None
    sent=0

    for row in df.iter_rows(named=True):
        current_timestamp=row['pickup_hour']

        if(prev_timestamp is not None and current_timestamp!=prev_timestamp):
            hours_elapsed=(
                current_timestamp-prev_timestamp
            ).total_seconds()/3600

            sleep_time=(
                hours_elapsed*3600/STREAM_REPLAY_SPEED
            )
            time.sleep(sleep_time)

        message=serialize_row(row)

        producer.send(KAFKA_TOPIC,message)

        sent+=1

        if sent % 10000 == 0:
            logger.info(
                f"Sent {sent} messages"
            )
            producer.flush()

        prev_timestamp=current_timestamp
    logger.info(f"Replay complete. Total messages sent: {sent}")

if __name__ == "__main__":
    replay_stream()