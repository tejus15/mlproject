import logging
import os
from datetime import datetime

def create_log_file(msg):           

    # Create log file name
    LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

    # Get the location where the log file will be saved
    logs_path=os.path.join(os.getcwd(), "logs", LOG_FILE)

    # Create the folder. If folder already exists, just add the new file without affecting the previous files (exist_ok=True).
    os.makedirs(logs_path, exist_ok=True)

    # Get the complete path to the log file
    LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE)

    # Generic format of a log
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    # Start logging on info level
    logging.info(msg)
