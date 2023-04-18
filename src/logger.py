import logging
from datetime import datetime
import os

from src.config.configuration import *

ROOT_DIR  = ROOT_DIR_KEY

log_dir = 'zomato_logs'
log_dir_path = os.path.join(ROOT_DIR,log_dir)

current_time_stamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

log_file_name = f'log_{current_time_stamp}.log'

os.makedirs(log_dir_path,exist_ok=True)

log_file_path = os.path.join(log_dir_path,log_file_name)

logging.basicConfig(filename=log_file_path,
                    filemode='w',
                    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO
)