from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

TAXI_TYPE = "yellow"
YEAR = 2023
TRAIN_MONTHS = [1, 2, 3, 4, 5, 6]
STREAM_MONTHS = [7, 8, 9, 10, 11, 12]

TARGET_COL = "trip_count"
LOCATION_COL = "PULocationID"
TIME_COL = "pickup_hour"

MODEL_NAME = "ride_demand_xgb"
RANDOM_SEED = 42

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
}
VALIDATION_SPLIT_DATE = "2023-05-01" 

API_HOST = "0.0.0.0"
API_PORT = 8000

KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "ride_events"
STREAM_REPLAY_SPEED = 50.0     
CONSUMER_WINDOW_SIZE = 500     

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "ride-demand-forecast"

DRIFT_SHARE_THRESHOLD = 0.3    
DRIFT_P_VALUE = 0.05
RETRAIN_WINDOW_DAYS = 30       

RMSE_IMPROVEMENT_THRESHOLD = 0.95 