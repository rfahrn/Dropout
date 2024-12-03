import os
import yaml
import logging

def setup_logging(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging_config = config.get('LOGGING', {})
    log_file = logging_config.get('LOG_FILE')
    log_directory = os.path.dirname(log_file)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, logging_config.get('LOG_LEVEL', 'INFO').upper(), 'DEBUG'),
        format=logging_config.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        datefmt=logging_config.get('LOG_DATE_FORMAT', '%Y-%m-%d %H:%M:%S'))
    return logging
