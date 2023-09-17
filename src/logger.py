import os
import logging
from datetime import datetime

LOG_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",LOG_file)
os.makedirs(log_path,exist_ok=True)

LOG_File_path =os.path.join(log_path,LOG_file)

logging.basicConfig(
    filename=LOG_File_path,
    format="[%(asctime)s %(lineno)d %(name)s - %(levelname)s - %(message)s]",
    level=logging.INFO
)

