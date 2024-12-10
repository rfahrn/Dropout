
import logging
import os
import yaml
import polars as pl


class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
    def get(self, key, default=None):
        return self.config.get(key, default)


class DataLoader:
    def __init__(self, config, nr_rows=None, exclude_product_type=None):
        self.data_path = config.get("DATA_PATH")
        self.processed_data_path = config.get("PROCESSED_DATA_PATH")
        self.target_column = config.get("TARGET_COLUMN")
        self.nr_rows = nr_rows
        self.exclude_product_type = exclude_product_type
 
        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_datasets(self, segment=None):
        logging.info("Loading datasets...")
        lazy_df = pl.scan_csv(f"{self.data_path}\ordcustprod.csv", has_header=True,infer_schema_length=5000,low_memory=True, try_parse_dates=True,encoding="utf8",
                              rechunk=False,truncate_ragged_lines=True,ignore_errors=True, n_rows=self.nr_rows)
        lazy_df = lazy_df.sort("cust_no", "transaction_dte")
        df = lazy_df.collect() 
        art_info = pl.read_csv(f"{self.data_path}\ART_INFO.csv", has_header=True)
        molecules = pl.read_csv(f"{self.data_path}\molecules.csv", has_header=True)
        if segment and segment.lower() != "all":
            logging.info(f"Filtering data for segment: {segment}")
            df = df.filter(pl.col("segment_col") == segment)
        if self.exclude_product_type:
            
            logging.info(f"Excluding Product Type: {self.exclude_product_type}...")
            excluded_customers = df.filter(pl.col("Product Type") == self.exclude_product_type)["cust_no"].unique()
            df = df.filter(~pl.col("cust_no").is_in(excluded_customers))
        
        art_info = art_info.with_columns(pl.col("PHARMACODE").cast(pl.Int64)).rename({"PHARMACODE": "official Pharmacode"})
        art_info_filtered = art_info.select(["ART_ANR", "ART_BASENAME", "ART_FULLNAME", "ART_FORM", "CONCT", "ART_MULTIPL", "official Pharmacode"])
        df = df.with_columns(pl.col("official Pharmacode").cast(pl.Int64))
        df = df.join(art_info_filtered, on="official Pharmacode", how="inner")

        molecules_filtered = molecules.select(["ART_ANR", "MolText", "MONO COMBI"])
        df = df.join(molecules_filtered, on="ART_ANR", how="left")
        logging.info("jointed with atc_prod and molecule data")
        complaints = pl.read_excel(f"{self.data_path}\MVS.xlsx", has_header=True)
        complaints = complaints.select(["cust_no","ord_no", "fault_description_txt","fault_dte","fault_category_1","fault_category_2","fault_category_3","fault_category_4","fault_category_5","fault_source_1"])
        complaints = complaints.with_columns(pl.col("cust_no").cast(pl.Int64))

        df = df.join(complaints, left_on=["cust_no", "ord_no"], right_on=["cust_no", "ord_no"], how="inner")
        logging.info("jointed with complaints data")
    
        logging.info("Data loaded successfully.")
        return df