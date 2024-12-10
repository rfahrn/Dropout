import os
import polars as pl
import logging

class DataLoader:
    def __init__(self, config):
        self.data_path = config.get("DATA_PATH")
        self.processed_data_path = config.get("PROCESSED_DATA_PATH")
        self.target_column = config.get("TARGET_COLUMN")
        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_datasets(self, segment=None):
        logging.info("Loading datasets...")
        lazy_df = pl.scan_csv(f"{self.data_path}/ordcustprod.csv", has_header=True, infer_schema_length=5000, low_memory=True, try_parse_dates=True, encoding="utf8", rechunk=False)
        lazy_df = lazy_df.sort("cust_no", "transaction_dte")
        df = lazy_df.collect()
        art_info = pl.read_csv(f"{self.data_path}/ART_INFO.csv", has_header=True)
        molecules = pl.read_csv(f"{self.data_path}/molecules.csv", has_header=True)
        
        if segment and segment.lower() != "all":
            logging.info(f"Filtering data for segment: {segment}")
            df = df.filter(pl.col("segment_col") == segment)
        
        logging.info("Data loaded successfully.")
        return df, art_info, molecules