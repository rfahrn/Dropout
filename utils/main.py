import os
import yaml
import logging
import logging_set
log = logging_set.setup_logging()


class DataLoader:
    def __init__(self, config):
        self.data_path = config.get("DATA_PATH")
        self.processed_data_path = config.get("PROCESSED_DATA_PATH")
        self.target_column = config.get("TARGET_COLUMN")
        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_datasets(self, segment=None):
        logging.info("Loading datasets...")
        logging.info("Data loaded successfully.")


if __name__ == "__main__":
    logging.info("Starting the main program...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    data_loader = DataLoader(config)
    logging.info("Finished running the main program.")
