import json
from datetime import datetime
from kafka import KafkaConsumer
from config import (
    KAFKA_BOOTSTRAP,KAFKA_TOPIC,
    CONSUMER_WINDOW_SIZE
)
from src.serving.feature_store import FeatureStore
from src.utils.logger import logger
from src.monitoring.drift_detector import check_drift
import traceback
featurestore=FeatureStore()

def get_consumer()->KafkaConsumer:
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v:json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",   # start from beginning of topic
        group_id="ride-demand-consumer",
        enable_auto_commit=True,
    )

def process_message(message:dict)->None:
    zone_id = message["zone_id"]
    timestamp = datetime.fromisoformat(message["pickup_hour"])
    trip_count = message["trip_count"]

    featurestore.set_trip_count(zone_id=zone_id,timestamp=timestamp,count=trip_count)

def consume(on_window_ready=None):
    consumer=get_consumer()
    batch=[]
    total=0
    current_week=None

    logger.info(f"Starting consumer on topic '{KAFKA_TOPIC}'")

    for message in consumer:
        data=message.value
        process_message(data)

        msg_dt=datetime.fromisoformat(data['pickup_hour'])
        msg_week=(msg_dt.year,msg_dt.isocalendar()[1])

        if current_week is None:
            current_week=msg_week

        if current_week!=msg_week:
            if on_window_ready and batch:
                logger.info(f"Week {current_week} window ready: {len(batch)} messages")
                on_window_ready(batch)

            batch=[]
            current_week=msg_week
        batch.append(data)


def on_window_ready(batch: list[dict]) -> None:
    should_retrain = check_drift(batch)
    if should_retrain:
        logger.info(200*"*")
        logger.info("retrain needed")
        logger.info(200*"*")

if __name__=='__main__':
    consume(on_window_ready=on_window_ready)





