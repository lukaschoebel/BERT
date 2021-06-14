import logging

def logger_factory(name: str, logging_level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s'))
    logger.addHandler(handler)

    return logger