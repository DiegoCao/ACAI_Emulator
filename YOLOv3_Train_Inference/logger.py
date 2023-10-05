# Created by Churong Ji at 10/05/23

import logging
import sys


def setup_logger(name, log_file, write_type, level=logging.INFO):
    """To set up as many loggers as you want"""

    formatter = logging.Formatter('%(message)s')
    if write_type == "log":
        formatter = logging.Formatter('%(asctime)s %(message)s')

    handler_1 = logging.FileHandler(log_file)
    handler_2 = logging.StreamHandler(sys.stdout)
    handler_1.setFormatter(formatter)
    handler_2.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler_1)
    if write_type == "log":
        logger.addHandler(handler_2)

    return logger
