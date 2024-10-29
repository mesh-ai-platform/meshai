# meshai/logger.py

import logging

def setup_logger(name='meshai_logger', log_file='meshai.log', level=logging.INFO):
    """
    Sets up a logger with the specified name and log file.
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
