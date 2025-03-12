import logging
import sys

# General purpose utility functions


def get_console_logger(name, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def suffix_iterator_items(iterator, suffix):
    for item in iterator:
        if item[-1] != suffix:
            item += suffix
        yield item
