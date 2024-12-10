import polars as pl
from Dropout.utils import clean_company_names, normalize_baseproducts
import logging

class Preprocessor:
    def __init__(self, max_columns=1000000):
        self.max_columns = max_columns

    def preprocess(self, df, art_info, molecules):
        logging.info("Starting preprocessing...")
        df = self.join_product_and_molecule_data(df, art_info, molecules)
        df = self.filter_and_sort_data(df)
        df = self.add_previous_transaction_features(df)
        df = self.calculate_additional_features(df)
        df = self.calculate_unique_products_and_affinity(df)
        df = self.calculate_median_dot_stats(df)
        df = self.calculate_adherence(df)
        df = self.clean_and_normalize_data(df)
        logging.info("Preprocessing completed.")
        return df

    def join_product_and_molecule_data(self, df, art_info, molecules):
        art_info = art_info.with_columns(pl.col("PHARMACODE").cast(pl.Int64)).rename({"PHARMACODE": "official Pharmacode"})
        art_info_filtered = art_info.select(["ART_ANR", "ART_BASENAME", "ART_FULLNAME", "ART_FORM", "CONCT", "ART_MULTIPL", "official Pharmacode"])
        df = df.with_columns(pl.col("official Pharmacode").cast(pl.Int64))
        df = df.join(art_info_filtered, on="official Pharmacode", how="inner")
        molecules_filtered = molecules.select(["ART_ANR", "MolText", "MONO COMBI"])
        df = df.join(molecules_filtered, on="ART_ANR", how="left")
        return df

    def filter_and_sort_data(self, df):
        df = df.filter(pl.col("Product ATC (WHO)").is_not_null() & pl.col("Product ATC (EPHMRA)").is_not_null())
        df = df.sort("cust_no", "transaction_dte")
        df = df.with_columns([pl.col("Product DOT/Unit PCG").fill_null(1.0)])
        df = df.with_columns(pl.when(pl.col("Product ATC (WHO)").is_not_null()).then(pl.col("Product ATC (EPHMRA)").str.slice(0, 6)).otherwise(pl.col("Product ATC (EPHMRA)").str.slice(0, 6)).alias("product_atc"))
        return df

    def add_previous_transaction_features(self, df):
        df = df.with_columns([
            pl.col("pharm_no").shift(1).over("cust_no").alias("prev_pharm_no"),
            pl.col("product_atc").shift(1).over("cust_no").alias("prev_product_atc"),
            pl.col("transaction_dte").shift(1).over("cust_no").alias("prev_transaction_dte"),
            pl.col("Mixed DOT").mean().over("cust_no").alias("mixed_mean_DOT")
        ])
        return df

    def calculate_additional_features(self, df):
        df = df.with_columns((pl.col("transaction_dte") - pl.col("prev_transaction_dte")).dt.total_days().alias("days_since_prev"))
        df = df.with_columns([
            (pl.col("days_since_prev") / pl.col("mixed_mean_DOT")).fill_null(0).alias("time_factor"),
            pl.when(pl.col("product_atc") != pl.col("prev_product_atc")).then(1).otherwise(0).alias("atc_similarity_factor"),
            pl.when(pl.col("pharm_no") != pl.col("prev_pharm_no")).then(1).otherwise(0).alias("product_change_factor")
        ])
        df = df.with_columns((pl.col("time_factor") + pl.col("atc_similarity_factor") + pl.col("product_change_factor")).alias("product_switch_affinity"))
        return df

    def calculate_unique_products_and_affinity(self, df):
        unique_products_per_customer = df.select(["cust_no", "product_atc", "pharm_no"]).unique(subset=["cust_no", "product_atc", "pharm_no"])
        product_counts = unique_products_per_customer.group_by("cust_no").agg(pl.count("product_atc").alias("total_products"))
        df = df.join(product_counts, on="cust_no", how="inner")
        product_affinity = df.group_by("cust_no").agg(pl.col("product_atc").n_unique().alias("unique_therapeutic_classes"), pl.col("product_atc").count().alias("total_products_bought")).with_columns(
            (1 - (1 / pl.col("unique_therapeutic_classes"))).alias("product_affinity_score")
        )
        df = df.join(product_affinity, on="cust_no", how="inner")
        return df

    def calculate_median_dot_stats(self, df):
        product_dot_stats = df.group_by(["pharm_no", "product_atc"]).agg(
            pl.col("Mixed DOT").median().alias("median_DOT_per_product"),
            pl.col("Mixed DOT").mean().alias("mean_DOT_per_product")
        )
        df = df.join(product_dot_stats, on=["pharm_no", "product_atc"], how="inner")
        df = df.with_columns([pl.col("median_DOT_per_product").shift(1).over("cust_no").alias("prev_median_DOT")])
        return df

    def calculate_adherence(self, df):
        df = df.with_columns(
            pl.when(pl.col("prev_median_DOT").is_not_null())
            .then((pl.col("days_since_prev") / pl.col("prev_median_DOT")).alias("refill_ratio"))
            .otherwise(None)
        )
        df = df.with_columns([
            pl.when(pl.col("refill_ratio") < 0.9).then(1).otherwise(0).alias("early_refill_flag"),
            pl.when(pl.col("refill_ratio") > 1.1).then(1).otherwise(0).alias("late_refill_flag"),
            pl.when(pl.col("refill_ratio").is_not_null()).then(1 / (1 + abs(pl.col("refill_ratio") - 1))).otherwise(None).alias("adherence_score")
        ])
        adherence_summary = df.group_by("cust_no").agg([
            pl.col("adherence_score").mean().alias("mean_adherence_score"),
            pl.col("early_refill_flag").sum().alias("total_early_refills"),
            pl.col("late_refill_flag").sum().alias("total_late_refills"),
            pl.col("refill_ratio").median().alias("median_refill_ratio"),
            pl.col("refill_ratio").std().alias("std_dev_refill_ratio"),
        ])
        df = df.join(adherence_summary, on="cust_no", how="inner")
        return df

    def clean_and_normalize_data(self, df):
        df = clean_company_names(df=df, column_name='Product Company')
        df = clean_company_names(df=df, column_name= 'Main Supplier')
        df = normalize_baseproducts(df=df, column_name='Product Basename')
        df = normalize_baseproducts(df=df, column_name='MolText')
        return df