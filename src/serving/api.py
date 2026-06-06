from fastapi import FastAPI,HTTPException
from src.serving.schema import PredictionRequest,PredictionResponse
from config import API_HOST,API_PORT,PROCESSED_DIR,CURRENT_ZONE_HOUR_AVG,ORIGINAL_ZONE_HOUR_AVG
import polars as pl
from src.serving.model_loader import ModelLoader
from src.serving.feature_store import FeatureStore
from src.features.schema import FEATURE_COLUMNS
import pandas as pd
from src.utils.logger import logger
from contextlib import asynccontextmanager

model_loader=ModelLoader()
featurestore=FeatureStore()

def _load_zone_hour_avg() -> dict:
    path = CURRENT_ZONE_HOUR_AVG if CURRENT_ZONE_HOUR_AVG.exists() else ORIGINAL_ZONE_HOUR_AVG
    df = pl.read_parquet(path)
    lookup = {}
    for row in df.iter_rows(named=True):
        lookup[(row["PULocationID"], row["hour_of_day"])] = row["zone_hour_avg"]
    logger.info(f"Loaded {len(lookup)} zone-hour averages from {path.name}")
    return lookup

ZONE_HOUR_AVG: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up - loading model")
    model_loader.load()
    logger.info("Model ready")
    ZONE_HOUR_AVG=_load_zone_hour_avg()
    yield
    logger.info("Shutting down")

app=FastAPI(
    title="Ride Demand ForeeCasting API",
    lifespan=lifespan
)

@app.post("/predict",response_model=PredictionResponse)
def predict(request:PredictionRequest):
    try:
        lag_features=featurestore.get_lag_features(
            request.zone_id,
            request.pickup_hour
        )

        hour_of_day=request.pickup_hour.hour
        day_of_week=request.pickup_hour.weekday()
        month=request.pickup_hour.month
        is_weekend=True if day_of_week>=5 else False
        is_rush_hour=True if hour_of_day in [7,8,9,17,18,19] else False

        zone_hour_avg = ZONE_HOUR_AVG.get(
            (request.zone_id, request.pickup_hour.hour),
            0.0  
        )
        
        features = {
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush_hour,
            **lag_features,
            "PULocationID": request.zone_id,
            "zone_hour_avg": zone_hour_avg,
        }
        

        X=pd.DataFrame([features])[FEATURE_COLUMNS]
        X["PULocationID"] = X["PULocationID"].astype("category")

        prediction = model_loader.predict(X)[0]
        prediction = max(0.0, prediction) 

        return PredictionResponse(
            zone_id=request.zone_id,
            pickup_hour=request.pickup_hour,
            predicted_demand=round(prediction, 2),
            model_version=model_loader.get_version()
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/reload-model")
def reload_model():
    model_loader.load()
    return {
        "status": "ok",
        "message": "Model Reloaded successfully"
    }
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": model_loader.get_version()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)