import torch
import numpy as np
import random
import logging
import sys
import datetime
from pytz import timezone, utc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)


def get_logger(name=None):
    if not name:
        name = 'main'
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    def customTime(*args):
        utc_dt = datetime.datetime.now()
        my_tz = timezone("Asia/Seoul")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()
    logging.Formatter.converter = customTime
    return logger
