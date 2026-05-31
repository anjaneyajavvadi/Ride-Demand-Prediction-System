import logging
import os
from datetime import datetime

LOG_DIR="logs"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE=os.path.join(
    LOG_DIR,
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
)

logger=logging.getLogger("ride-analysis")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler=logging.FileHandler(LOG_FILE,encoding="utf-8")
    formatter=logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

