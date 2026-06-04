from pydantic import BaseModel,Field
from datetime import datetime


class PredictionRequest(BaseModel):
    zone_id: int = Field(..., ge=1, le=263)
    pickup_hour: datetime

class PredictionResponse(BaseModel):
    zone_id: int
    pickup_hour: datetime
    predicted_demand: float
    model_version: str