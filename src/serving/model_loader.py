import mlflow.xgboost
import xgboost as xgb
from threading import Lock
from config import MLFLOW_TRACKING_URI,MODEL_NAME
import mlflow
from src.utils.logger import logger

class ModelLoader:
    def __init__(self):
        self._model=None
        self._model_version=None
        self._lock=Lock()

    def load(self,model_name:str='ride_demand_xgboost',stage:str='latest'):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{model_name}/{stage}"
        model=mlflow.xgboost.load_model(
            model_uri
        )

        with self._lock:
            self._model=model
            self._model_version=stage

        logger.info(f"Loaded model '{model_name}' stage '{stage}'")


    def predict(self,X)->list[float]:
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not loaded")
            return self._model.predict(X).tolist()

    def get_version(self)->str:
        return self._model_version or "unknown"
    
    
        


