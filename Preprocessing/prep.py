import polars as pl
from utils import clean_company_names, normalize_baseproducts
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from typing import Optional, List, Dict, Any

class Preprocessor:
    """
    A flexible Preprocessor class to clean, filter, and engineer features from transaction-level data.
    
    Parameters
    ----------
    atc_columns : List[str], optional
        List of ATC-related columns to filter and slice.
    time_window_days : int, optional
        Default time window for recent purchases (default=180 days).
    max_columns : int, optional
        Maximum number of columns to process.
    """

    def __init__(self,atc_columns: Optional[List[str]] = None,time_window_days: int = 180,max_columns: int = 1_000_000):
        self.atc_columns = atc_columns or ["Product ATC (WHO)", "Product ATC (EPHMRA)"]
        self.time_window_days = time_window_days
        self.max_columns = max_columns
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Run the full preprocessing pipeline.
        """
        print("Starting preprocessing...")
        
        df = self.filter_and_sort_data(df)
        df = self.add_transaction_features(df)
        df = self.calculate_customer_metrics(df)
        df = self.add_adherence_metrics(df)
        df = self.calculate_product_affinity(df)
        df = self.normalize_and_clean_data(df)
        df = self.encode_categorical_columns(df)

        #df = self.add_previous_transaction_features(df)
        #df = self.calculate_additional_features(df)
        #df = self.calculate_unique_products_and_affinity(df)
        #df = self.calculate_median_dot_stats(df)
        #df = self.calculate_adherence(df)
        #df = self.clean_and_normalize_data(df)
        #df = self.encode_categorical_columns(df)

        print("Preprocessing completed.")
        return df

    def filter_and_sort_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter out rows without product ATC codes and sort by customer and transaction date.
        Also, generate a short product_atc code column.
        """
        return (
            df.filter(
                pl.col(self.atc_columns[0]).is_not_null() &
                pl.col(self.atc_columns[1]).is_not_null()
            )
            .sort(["cust_no", "transaction_dte"])
            .with_columns([
                pl.col("Product DOT/Unit PCG").fill_null(1.0),
                pl.col(self.atc_columns[1]).str.slice(0, 6).alias("product_atc")
            ])
        )
    
    def add_transaction_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add transaction-level features."""
        return df.with_columns([
            pl.col("transaction_dte").shift(1).over("cust_no").alias("prev_transaction_dte"),
            (pl.col("transaction_dte") - pl.col("prev_transaction_dte")).dt.total_days()
            .alias("days_since_prev"),
            pl.col("Mixed DOT").mean().over("cust_no").alias("mean_mixed_DOT"),
        ])
    
    def calculate_customer_metrics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate RFM and interorder metrics."""
        return df.group_by("cust_no").agg([
            pl.col("transaction_dte").max().alias("last_purchase"),
            (pl.col("transaction_dte").max() - pl.col("transaction_dte").min()).dt.total_days()
            .alias("purchase_duration_days"),
            pl.count("ord_no").alias("order_frequency"),
            pl.col("Mixed DOT").mean().alias("avg_mixed_DOT"),
        ])
    
    def add_adherence_metrics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Compute adherence scores and refill behavior."""
        df = df.with_columns(
            pl.when(pl.col("days_since_prev").is_not_null())
            .then(pl.col("days_since_prev") / pl.col("mean_mixed_DOT"))
            .otherwise(None)
            .alias("refill_ratio")
        )

        return df.with_columns([
            pl.when(pl.col("refill_ratio") < 0.9).then(1).otherwise(0).alias("early_refill"),
            pl.when(pl.col("refill_ratio") > 1.1).then(1).otherwise(0).alias("late_refill"),
            (1 / (1 + (pl.col("refill_ratio") - 1).abs())).alias("adherence_score"),
        ])
    
    def calculate_product_affinity(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate product diversity and affinity scores."""
        product_stats = df.group_by("cust_no").agg([
            pl.col("product_atc").n_unique().alias("unique_product_classes"),
            pl.col("product_atc").count().alias("total_product_count"),
        ])
        return df.join(product_stats, on="cust_no").with_columns(
            (1 - (1 / pl.col("unique_product_classes"))).alias("product_affinity_score")
        )

    def normalize_and_clean_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Clean and normalize textual columns."""
        if "Product Company" in df.columns:
            df = clean_company_names(df, "Product Company")
        if "Product Basename" in df.columns:
            df = normalize_baseproducts(df, "Product Basename")
        return df

    def encode_categorical_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Encode categorical columns using LabelEncoder."""
        categorical_columns = ["dominant_refill_trend", "product_atc"]

        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                col_values = df.select(pl.col(col)).collect().to_series()
                encoded = le.fit_transform(col_values.to_list())
                self.label_encoders[col] = le
                df = df.with_columns(pl.Series(name=f"{col}_encoded", values=encoded))
        return df


    def add_previous_transaction_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add features related to the previous transaction of the same customer, such as:
        - Previous pharm_no
        - Previous product_atc
        - Previous transaction date
        - Mean Mixed DOT over customer history
        """
        df = df.with_columns([
            pl.col("pharm_no").shift(1).over("cust_no").alias("prev_pharm_no_over_cust_no"),
            pl.col("product_atc").shift(1).over("cust_no").alias("prev_product_atc_over_cust_no"),
            pl.col("transaction_dte").shift(1).over("cust_no").alias("prev_transaction_dte_over_cust_no"),
            pl.col("Mixed DOT").mean().over("cust_no").alias("mixed_mean_DOT_over_cust_no")
        ])
        return df

    def calculate_additional_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate additional time and similarity-based features:
        - days_since_prev_cust_no: days between current and previous transaction of same customer
        - time_factor: ratio of days_since_prev to mean DOT
        - atc_similarity_factor: indicates if current product_atc differs from previous product_atc
        - product_change_factor: indicates if current pharm_no differs from previous pharm_no
        - product_switch_affinity: composite score based on time_factor, atc_similarity_factor, product_change_factor
        """
        df = df.with_columns(
            (pl.col("transaction_dte") - pl.col("prev_transaction_dte_over_cust_no"))
            .dt.total_days()
            .alias("days_since_prev_cust_no")
        )

        df = df.with_columns([
            (pl.col("days_since_prev_cust_no") / pl.col("mixed_mean_DOT_over_cust_no"))
            .fill_null(0)
            .alias("time_factor"),
            pl.when(pl.col("product_atc") != pl.col("prev_product_atc_over_cust_no"))
            .then(1).otherwise(0)
            .alias("atc_similarity_factor"),
            pl.when(pl.col("pharm_no") != pl.col("prev_pharm_no_over_cust_no"))
            .then(1).otherwise(0)
            .alias("product_change_factor")
        ])

        df = df.with_columns(
            (pl.col("time_factor") + pl.col("atc_similarity_factor") + pl.col("product_change_factor")).alias("product_switch_affinity"))

        return df

    def calculate_unique_products_and_affinity(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate unique products and product affinity per customer:
        - total_products: total unique (pharm_no, product_atc) combinations per customer
        - unique_therapeutic_classes: number of unique product_atc classes per customer
        - product_affinity_score: a derived score based on the number of unique therapeutic classes
        """
        # Unique combos per customer
        unique_products = df.select(["cust_no", "product_atc", "pharm_no"]).unique(subset=["cust_no", "product_atc", "pharm_no"])
        product_counts = unique_products.group_by("cust_no").agg(pl.count("product_atc").alias("total_products"))
        df = df.join(product_counts, on="cust_no", how="inner")

        # Therapeutic classes and affinity
        product_affinity = df.group_by("cust_no").agg(
            pl.col("product_atc").n_unique().alias("unique_therapeutic_classes"),
            pl.col("product_atc").count().alias("total_products_bought")
        ).with_columns(
            (1 - (1 / pl.col("unique_therapeutic_classes"))).alias("product_affinity_score")
        )

        df = df.join(product_affinity, on="cust_no", how="inner")
        return df

    def calculate_median_dot_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate median and mean DOT per product and join these stats back to the main DataFrame.
        Also adds 'prev_median_DOT' for the last product the customer used.
        """
        # Product-level median/mean DOT
        product_dot_stats = df.group_by(["pharm_no", "product_atc"]).agg(
            pl.col("Mixed DOT").median().alias("median_DOT_per_product"),
            pl.col("Mixed DOT").mean().alias("mean_DOT_per_product")
        )

        df = df.join(product_dot_stats, on=["pharm_no", "product_atc"], how="inner")

        # Previous product median DOT for refill calculations
        df = df.with_columns(
            pl.col("median_DOT_per_product").shift(1).over("cust_no").alias("prev_median_DOT")
        )
        return df

    def calculate_adherence(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate adherence metrics and refill behavior:
        - refill_ratio: ratio of days_since_prev to prev_median_DOT
        - early/late refill flags
        - adherence_score: derived from refill_ratio
        Aggregates these metrics per customer and calculates composite adherence measures.
        """
        # Refill ratio and adherence
        df = df.with_columns(
            pl.when(pl.col("prev_median_DOT").is_not_null())
            .then(pl.col("days_since_prev_cust_no") / pl.col("prev_median_DOT"))
            .otherwise(None)
            .alias("refill_ratio")
        )

        df = df.with_columns([
            pl.when(pl.col("refill_ratio") < 0.9).then(1).otherwise(0).alias("early_refill_flag"),
            pl.when(pl.col("refill_ratio") > 1.1).then(1).otherwise(0).alias("late_refill_flag"),
            pl.when(pl.col("refill_ratio").is_not_null())
            .then(1 / (1 + abs(pl.col("refill_ratio") - 1)))
            .otherwise(None)
            .alias("adherence_score")
        ])

        # Per-customer adherence summary
        adherence_summary = df.group_by("cust_no").agg([
            pl.col("adherence_score").mean().alias("mean_adherence_score"),
            pl.col("early_refill_flag").sum().alias("total_early_refills"),
            pl.col("late_refill_flag").sum().alias("total_late_refills"),
            pl.col("refill_ratio").median().alias("median_refill_ratio"),
            pl.col("refill_ratio").std().alias("std_dev_refill_ratio"),
        ])
        df = df.join(adherence_summary, on="cust_no", how="inner")

        # Mean refill gap calculations (overall)
        mean_refill_gap = df.group_by(["cust_no", "pharm_no", "product_atc"]).agg(
            pl.col("days_since_prev_cust_no").mean().alias("mean_refill_gap_overall")
        )

        # Last 8 months filter
        most_recent_date = df.select(pl.col("transaction_dte").max())[0,0]
        last_8_months_df = df.filter(
            pl.col("transaction_dte") >= (most_recent_date - timedelta(days=240))
        )
        mean_refill_gap_recent = last_8_months_df.group_by(["cust_no", "pharm_no", "product_atc"]).agg(
            pl.col("days_since_prev_cust_no").mean().alias("mean_refill_gap_recent")
        )

        # Join recent and overall metrics
        df = df.join(mean_refill_gap, on=["cust_no", "pharm_no", "product_atc"], how="left")
        df = df.join(mean_refill_gap_recent, on=["cust_no", "pharm_no", "product_atc"], how="left")

        # Compare recent vs. overall gaps -> refill trend
        df = df.with_columns(
            pl.when(pl.col("mean_refill_gap_recent") < pl.col("mean_refill_gap_overall")).then(1)
            .when(pl.col("mean_refill_gap_recent") > pl.col("mean_refill_gap_overall")).then(-1)
            .otherwise(0).alias("refill_trend")
        )

        # Composite adherence score
        df = df.with_columns(
            (pl.col("mean_adherence_score") * 
             (1 / (1 + abs(pl.col("mean_refill_gap_recent") - pl.col("mean_refill_gap_overall")))))
            .alias("composite_adherence_score")
        )

        # Customer-level refill trend summary
        customer_summary = df.group_by("cust_no").agg([
            pl.col("composite_adherence_score").mean().alias("mean_composite_adherence_score"),
            pl.col("refill_trend").value_counts().alias("refill_trend_summary")
        ])

        df = df.join(customer_summary, on="cust_no", how="inner")

        # Explode and pivot trend summary
        exploded = df.explode("refill_trend_summary").filter(pl.col("refill_trend_summary").is_not_null())
        exploded = exploded.with_columns([
            pl.col("refill_trend_summary").struct.field("refill_trend").alias("trend_category"),
            pl.col("refill_trend_summary").struct.field("count").alias("trend_counts")
        ]).drop("refill_trend_summary")

        trend_counts = exploded.group_by(["cust_no", "trend_category"]).agg(
            pl.col("trend_counts").sum().alias("total_counts")
        )

        trend_counts = trend_counts.with_columns(
            pl.when(pl.col("trend_category") == 1).then(pl.lit("Improving"))
            .when(pl.col("trend_category") == 0).then(pl.lit("Stable"))
            .when(pl.col("trend_category") == -1).then(pl.lit("Declining"))
            .otherwise(pl.lit("Unknown"))
            .alias("trend_category_label")
        )

        trend_counts_pivot = trend_counts.pivot(
            values="total_counts",
            index=["cust_no"],
            columns=["trend_category_label"]
        ).fill_null(0)

        # Ensure columns exist
        for col_ in ["Improving", "Stable", "Declining"]:
            if col_ not in trend_counts_pivot.columns:
                trend_counts_pivot = trend_counts_pivot.with_columns(pl.lit(0).alias(col_))

        df = df.join(trend_counts_pivot, on="cust_no", how="inner").fill_null(0)

        # Calculate proportions and dominant trend
        df = df.with_columns([
            (pl.col("Improving") / (pl.col("Improving") + pl.col("Stable") + pl.col("Declining")))
            .fill_null(0).alias("proportion_improving"),
            (pl.col("Stable") / (pl.col("Improving") + pl.col("Stable") + pl.col("Declining")))
            .fill_null(0).alias("proportion_stable"),
            (pl.col("Declining") / (pl.col("Improving") + pl.col("Stable") + pl.col("Declining")))
            .fill_null(0).alias("proportion_declining")
        ])

        df = df.with_columns(
            pl.when((pl.col("Improving") > pl.col("Stable")) & (pl.col("Improving") > pl.col("Declining")))
            .then("Improving")
            .when(pl.col("Declining") > pl.col("Stable"))
            .then("Declining")
            .otherwise("Stable")
            .alias("dominant_refill_trend")
        )

        # Now 'dominant_refill_trend' is still a string, so we can safely do string comparisons:
        df = df.with_columns([
            pl.when(pl.col("dominant_refill_trend") == "Improving").then(1).otherwise(0).alias("improving_flag"),
            pl.when(pl.col("dominant_refill_trend") == "Declining").then(1).otherwise(0).alias("declining_flag"),
            pl.when(pl.col("dominant_refill_trend") == "Stable").then(1).otherwise(0).alias("stable_flag"),
        ])
        df = df.with_columns(pl.col("dominant_refill_trend").cast(pl.Float32))


        df = df.with_columns(
            (pl.col("Improving") / (pl.col("Declining") + 1)).alias("itd_ratio")
        )

        df = df.with_columns(
            (1.0 * pl.col("proportion_declining") + 0.5 * pl.col("proportion_stable") - 0.5 * pl.col("proportion_improving"))
            .alias("adherence_risk_score")
        )

        df = df.with_columns(
            (pl.col("Mixed DOT") - pl.col("mixed_mean_DOT_over_cust_no")).alias("normalized_mixed_DOT")
        )

        df = df.with_columns([
            pl.when(pl.col("dominant_refill_trend") == "Improving").then(1).otherwise(0).alias("improving_flag"),
            pl.when(pl.col("dominant_refill_trend") == "Declining").then(1).otherwise(0).alias("declining_flag"),
            pl.when(pl.col("dominant_refill_trend") == "Stable").then(1).otherwise(0).alias("stable_flag"),
        ])

        trend_counts = df.group_by("cust_no").agg([
            pl.col("improving_flag").sum().alias("num_improving_products"),
            pl.col("declining_flag").sum().alias("num_declining_products"),
            pl.col("stable_flag").sum().alias("num_stable_products"),
        ])
        df = df.join(trend_counts, on="cust_no", how="inner")

        customer_features = df.group_by("cust_no").agg([
            pl.col("num_improving_products").first(),
            pl.col("num_declining_products").first(),
            pl.col("num_stable_products").first(),
            pl.col("adherence_risk_score").mean().alias("avg_adherence_risk_score"),
        ])
        df = df.join(customer_features, on="cust_no", how="inner")

        return df
    

    def clean_and_normalize_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and normalize textual columns using defined functions. 
        Also demonstrates how to apply label encoding to categorical columns if needed.
        """
        # Clean company names
        if "Product Company" in df.columns:
            df = clean_company_names(df, "Product Company")
        if "Main Supplier" in df.columns:
            df = clean_company_names(df, "Main Supplier")

        # Normalize base products
        if "Product Basename" in df.columns:
            df = normalize_baseproducts(df, "Product Basename")
        if "MolText" in df.columns:
            df = normalize_baseproducts(df, "MolText")

        # Example of label encoding a categorical column (e.g., "dominant_refill_trend")
        # Adjust or add columns as needed:
        categorical_columns = ["dominant_refill_trend"]
        for col in categorical_columns:
            if col in df.columns:
                # Extract column as list and fit LabelEncoder
                le = LabelEncoder()
                cat_values = df[col].to_list()
                cat_values = ["Unknown" if v is None else v for v in cat_values]  # handle None
                encoded = le.fit_transform(cat_values)
                df = df.with_columns(pl.Series(name=col, values=encoded))

        return df
