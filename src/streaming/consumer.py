import json
from datetime import datetime
from kafka import KafkaConsumer
from config import (
    KAFKA_BOOTSTRAP,KAFKA_TOPIC,
    CONSUMER_WINDOW_SIZE
)
from src.serving.feature_store import FeatureStore
from src.utils.logger import logger

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

    logger.info(f"Starting consumer on topic '{KAFKA_TOPIC}'")

    for message in consumer:
        data=message.value
        process_message(data)
        batch.append(data)
        total+=1

        if(len(batch)>=CONSUMER_WINDOW_SIZE):
            logger.info(f"Window ready: {len(batch)} messages")

            try:
                on_window_ready(batch)
            except Exception as e:
                logger.error(
                    f"Window callback failed: {e}"
                )
            batch.clear()
        

        if total % 10000 == 0:
            logger.info(f"Consumed {total} messages total")

if __name__=='__main__':
    consume()





