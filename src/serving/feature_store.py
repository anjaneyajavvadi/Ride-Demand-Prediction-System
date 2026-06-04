import redis
from datetime import datetime,timedelta
from config import REDIS_HOST,REDIS_PORT,REDIS_DB,REDIS_TTL_SECONDS
from src.utils.logger import logger
import statistics

class FeatureStore:

    def __init__(self):
        self.client=redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )

    def _key(self,zone_id:int,timestamp:datetime)->str:
        ts=timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        return f"zone:{zone_id}:{ts}"

    def set_trip_count(self, zone_id: int, timestamp: datetime, count: int) -> None:
        key=self._key(zone_id,timestamp)
        self.client.setex(key,REDIS_TTL_SECONDS,count)

    def get_trip_count(self,zone_id:int,timestamp:datetime)->float:
        key=self._key(zone_id,timestamp)
        val=self.client.get(key)

        if val is None: 
            logger.warning(f"No data exists in cache for {zone_id} {timestamp}")
            return 0.0
        return float(val)
        
    def get_lag_features(self, zone_id: int, pickup_hour: datetime) -> dict:
        pipe = self.client.pipeline()
        for i in range(1, 169):
            pipe.get(self._key(zone_id, pickup_hour - timedelta(hours=i)))
        results=pipe.execute()

        counts=[float(v) if v is not None else 0.0 for v in results]

        lag_1h=counts[0]
        lag_24h=counts[23]
        lag_168h=counts[167]

        rolling_mean_3h=sum(counts[:3])/3
        rolling_mean_24h=sum(counts[:24])/24
        rolling_mean_168h=sum(counts[:168])/168

        rolling_std_24h  = statistics.stdev(counts[:24])  if len(counts[:24])  > 1 else 0.0
        rolling_std_168h = statistics.stdev(counts[:168]) if len(counts[:168]) > 1 else 0.0

        return {
            "lag_1h": lag_1h,
            "lag_24h": lag_24h,
            "lag_168h": lag_168h,
            "rolling_mean_3h": rolling_mean_3h,
            "rolling_mean_24h": rolling_mean_24h,
            "rolling_mean_168h":rolling_mean_168h,
            'rolling_std_24h':rolling_std_24h,
            'rolling_std_168h':rolling_std_168h
        }




        

        