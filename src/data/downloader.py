import os
import requests
from src.utils.logger import logger
from config import RAW_DIR,DATA_URL,YEAR,TAXI_TYPE


def download_data():
    logger.info(f"Started downloading data of year: {YEAR}")
    if not os.path.exists(RAW_DIR):
        logger.info(f"{RAW_DIR} does not exist. creating a new directory")
        os.makedirs(RAW_DIR, exist_ok=True)


    for month in range(1,13):
        logger.info(f"Downloading month:{month:02d}")
        url=DATA_URL.format(month=month,type=TAXI_TYPE,year=YEAR)
        file_path=RAW_DIR/f"{TAXI_TYPE}_tripdata_{YEAR}-{month:02d}.parquet"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Downloaded month {month:02d}")

        except Exception as e:
            logger.error(f"Failed month {month:02d}: {e}")


if __name__=="__main__":
    download_data()
        


    
