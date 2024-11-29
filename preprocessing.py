# upload file from one folder above 
import os
import sys
import polars as pl
from utils import clean_company_names, normalize_baseproducts, ColumnVectorizer



# Import libraries
import os
import yaml
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Set paths from config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["DATA_PATH"]
PROCESSED_DATA_PATH = config["PROCESSED_DATA_PATH"]


# Ensure output directory exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


def load_data():
    lazy_df = pl.scan_csv(f"{DATA_PATH}\ordcustprod.csv",has_header=True,
    infer_schema_length=5000, 
    low_memory=True,
    try_parse_dates=True,
    encoding="utf8",        
    rechunk=False   ,
    truncate_ragged_lines=True,
    ignore_errors=True)
    lazy_df = lazy_df.sort("cust_no", "transaction_dte")
    df = lazy_df.collect() # Load the data into memory
    art_info = pl.read_csv(f"{DATA_PATH}/ART_INFO.csv")
    molecules = pl.read_csv(f"{DATA_PATH}/molecules.csv")
    return df, art_info, molecules


# Clean and preprocess data
def preprocess_data(df, art_info, molecules):
    # Join additional information
    art_info = art_info.with_columns(pl.col("PHARMACODE").cast(pl.Int64)).rename({"PHARMACODE": "official Pharmacode"})
    art_info_filtered = art_info.select(["ART_ANR", "ART_BASENAME", "ART_FULLNAME", "ART_FORM", "CONCT", "ART_MULTIPL",  "official Pharmacode"])
    df2 = df.with_columns(pl.col("official Pharmacode").cast(pl.Int64))
    df2 = df2.join(art_info_filtered, on="official Pharmacode", how="inner")
    molecules_filtered = molecules.select(["ART_ANR", "MolText","MONO COMBI"]) 
    df2 = df2.join(molecules_filtered, on="ART_ANR", how="left")
    df2 = df2.filter(pl.col("Product ATC (WHO)").is_not_null() & pl.col("Product ATC (EPHMRA)").is_not_null())


    df2 = df2.sort("cust_no", "transaction_dte")
    df2 = df2.with_columns([pl.col("Product DOT/Unit PCG").fill_null(1.0)])

    df2 = df2.with_columns(pl.when(pl.col("Product ATC (WHO)").is_not_null()).then(pl.col("Product ATC (EPHMRA)").str.slice(0, 6)).otherwise(pl.col("Product ATC (EPHMRA)").str.slice(0, 6)).alias("product_atc"))

    # Add previous transaction details within the same customer group
    df2 = df2.with_columns([
        pl.col("pharm_no").shift(1).over("cust_no").alias("prev_pharm_no"),
        pl.col("product_atc").shift(1).over("cust_no").alias("prev_product_atc"),
        pl.col("transaction_dte").shift(1).over("cust_no").alias("prev_transaction_dte"),
        pl.col("Mixed DOT").mean().over("cust_no").alias("mixed_mean_DOT")])

    # Calculate the difference in days between transactions
    df2 = df2.with_columns((pl.col("transaction_dte") - pl.col("prev_transaction_dte")).dt.total_days().alias("days_since_prev"))

    # Calculate affinity components
    df2 = df2.with_columns([
        # Time Factor
        (pl.col("days_since_prev") / pl.col("mixed_mean_DOT")).fill_null(0).alias("time_factor"),
        # ATC Similarity Factor
        pl.when(pl.col("product_atc") != pl.col("prev_product_atc"))
        .then(1)
        .otherwise(0)
        .alias("atc_similarity_factor"),
        # Product Change Factor
        pl.when(pl.col("pharm_no") != pl.col("prev_pharm_no"))
        .then(1)
        .otherwise(0)
        .alias("product_change_factor")])
    # calculate affinity
    df2 = df2.with_columns((pl.col("time_factor") + pl.col("atc_similarity_factor") + pl.col("product_change_factor")).alias("product_switch_affinity"))
    print(len(df2))
    print(df2.shape)

    #  Calculate unique products per customer
    unique_products_per_customer = df2.select(["cust_no", "product_atc", "pharm_no"]).unique(subset=["cust_no", "product_atc", "pharm_no"])
    product_counts = unique_products_per_customer.group_by("cust_no").agg(pl.count("product_atc").alias("total_products"))

    df2 = df2.join(product_counts, on="cust_no", how="inner")
    print(len(df2))
    print(df2.shape)
    product_affinity = df2.group_by("cust_no").agg(pl.col("product_atc").n_unique().alias("unique_therapeutic_classes"),pl.col("product_atc").count().alias("total_products_bought")).with_columns(
        (1 - (1 / pl.col("unique_therapeutic_classes"))).alias("product_affinity_score")
    )
    df2 = df2.join(product_affinity, on="cust_no", how="inner")
    print(len(df2))
    print(df2.shape)

    # Calculate median DOT for each product (pharm_no + product_atc)
    product_dot_stats = df2.group_by(["pharm_no", "product_atc"]).agg(
        pl.col("Mixed DOT").median().alias("median_DOT_per_product"),
        pl.col("Mixed DOT").mean().alias("mean_DOT_per_product"))

    # Join back to the main DataFrame
    df2 = df2.join(product_dot_stats, on=["pharm_no", "product_atc"], how="inner")
    print(len(df2))
    print(df2.shape)
    # Add previous transaction details
    df2 = df2.with_columns([pl.col("median_DOT_per_product").shift(1).over("cust_no").alias("prev_median_DOT")])

    # Calculate refill adherence
    df2 = df2.with_columns(
        pl.when(pl.col("prev_median_DOT").is_not_null())
        .then((pl.col("days_since_prev") / pl.col("prev_median_DOT")).alias("refill_ratio"))
        .otherwise(None))
    df2 = df2.with_columns([
        pl.when(pl.col("refill_ratio") < 0.9).then(1).otherwise(0).alias("early_refill_flag"), # # Flag early refills (overlap)
        pl.when(pl.col("refill_ratio") > 1.1).then(1).otherwise(0).alias("late_refill_flag"),# Flag late refills (gap)
        # Adherence score (1 = perfect adherence, decrease for early/late refills)
        pl.when(pl.col("refill_ratio").is_not_null())
        .then(1 / (1 + abs(pl.col("refill_ratio") - 1)))
        .otherwise(None).alias("adherence_score")])

    adherence_summary = df2.group_by("cust_no").agg([
        pl.col("adherence_score").mean().alias("mean_adherence_score"),
        pl.col("early_refill_flag").sum().alias("total_early_refills"),
        pl.col("late_refill_flag").sum().alias("total_late_refills"),
        pl.col("refill_ratio").median().alias("median_refill_ratio"),
        pl.col("refill_ratio").std().alias("std_dev_refill_ratio"),
    ])
    print(len(df2))
    print(df2.shape)
    df2 = df2.join(adherence_summary, on="cust_no", how="inner")


    # Step 1: Calculate mean refill gaps per customer and product
    mean_refill_gap = df2.group_by(["cust_no", "pharm_no", "product_atc"]).agg(pl.col("days_since_prev").mean().alias("mean_refill_gap_overall"))

    # Step 2: Focus on transactions in the last 8 months
    most_recent_date =  df2.select(pl.col("transaction_dte").max())[0, 0]

    # Filter for last 8 months
    last_8_months_df = df2.filter(pl.col("transaction_dte") >= (most_recent_date - timedelta(days=240)))

    # Calculate mean refill gap for last 8 months
    mean_refill_gap_recent = last_8_months_df.group_by(["cust_no", "pharm_no", "product_atc"]).agg(pl.col("days_since_prev").mean().alias("mean_refill_gap_recent"))



    print(df2.shape)
    # Step 3: Join recent and overall metrics
    df2 = df2.join(mean_refill_gap, on=["cust_no", "pharm_no", "product_atc"], how="left")
    df2 = df2.join(mean_refill_gap_recent, on=["cust_no", "pharm_no", "product_atc"], how="left")
    print(len(df2))
    print(df2.shape)
    # Step 4: Compare recent vs. overall mean refill gaps
    df2 = df2.with_columns(
        pl.when(pl.col("mean_refill_gap_recent") < pl.col("mean_refill_gap_overall"))
        .then(1)  # Improving
        .when(pl.col("mean_refill_gap_recent") > pl.col("mean_refill_gap_overall"))
        .then(-1)  # Declining
        .otherwise(0)  # Stable
        .alias("refill_trend"))
    # Step 5: Calculate adherence score (composite metric)
    df2 = df2.with_columns(
        (pl.col("mean_adherence_score") * (1 / (1 + abs(pl.col("mean_refill_gap_recent") - pl.col("mean_refill_gap_overall"))))).alias("composite_adherence_score"))

    # Step 6: Summarize by customer
    customer_summary = df2.group_by("cust_no").agg([
        pl.col("composite_adherence_score").mean().alias("mean_composite_adherence_score"),
        pl.col("refill_trend").value_counts().alias("refill_trend_summary")
    ])

    # Step 7: Join back to the main DataFrame
    df2 = df2.join(customer_summary, on="cust_no", how="inner")


    print(len(df2))
    print(df2.shape)
    # Step 1: Explode refill trend summary
    exploded = df2.explode("refill_trend_summary").filter(pl.col("refill_trend_summary").is_not_null())


    # Step 2: Extract fields from struct
    exploded = exploded.with_columns([
        pl.col("refill_trend_summary").struct.field("refill_trend").alias("trend_category"),
        pl.col("refill_trend_summary").struct.field("count").alias("trend_counts")
    ]).drop("refill_trend_summary")

    # Step 3: Filter and aggregate counts by trend
    trend_counts = exploded.group_by(["cust_no", "trend_category"]).agg(
        pl.col("trend_counts").sum().alias("total_counts")
    )

    # Step 4: Map trend_category to descriptive labels
    trend_counts = trend_counts.with_columns(
        pl.when(pl.col("trend_category") == 1).then(pl.lit("Improving"))
        .when(pl.col("trend_category") == 0).then(pl.lit("Stable"))
        .when(pl.col("trend_category") == -1).then(pl.lit("Declining"))
        .otherwise(pl.lit("Unknown"))
        .alias("trend_category_label")
    )

    trend_counts_pivot = trend_counts.pivot(
        values="total_counts",  # Values to pivot
        index=["cust_no"],      # Group by customer
        columns=["trend_category_label"],  # Pivot on the label
        aggregate_function=None  # No aggregation needed
    )
    trend_counts_pivot = trend_counts_pivot.fill_null(0)
    # Ensure missing columns are added to the pivoted DataFrame
    for col in ["Improving", "Stable", "Declining"]:
        if col not in trend_counts_pivot.columns:
            trend_counts_pivot = trend_counts_pivot.with_column(pl.lit(0).alias(col))
    # Join pivoted data back to df2
    df2 = df2.join(trend_counts_pivot, on="cust_no", how="inner")
    df2 = df2.fill_null(0)
    df2 = df2.drop("refill_trend_summary")

    df2.head(2)
    df2 = df2.with_columns([
        # Proportions
        (pl.col("Improving") / (pl.col("Improving") + pl.col("Stable") + pl.col("Declining"))).fill_null(0).alias("proportion_improving"),
        (pl.col("Stable") / (pl.col("Improving") + pl.col("Stable") + pl.col("Declining"))).fill_null(0).alias("proportion_stable"),
        (pl.col("Declining") / (pl.col("Improving") + pl.col("Stable") + pl.col("Declining"))).fill_null(0).alias("proportion_declining")])
    df2 = df2.with_columns(
        pl.when(
            (pl.col("Improving") > pl.col("Stable")) &
            (pl.col("Improving") > pl.col("Declining"))
        ).then("Improving")
        .when(
            pl.col("Declining") > pl.col("Stable")
        ).then("Declining")
        .otherwise("Stable")
        .alias("dominant_refill_trend")
    )
    df2 = df2.with_columns(
        (pl.col("Improving") / (pl.col("Declining") + 1)).alias("itd_ratio")
    )
    df2 = df2.with_columns((1.0 * pl.col("proportion_declining") +0.5 * pl.col("proportion_stable") -0.5 * pl.col("proportion_improving")).alias("adherence_risk_score"))

    print(df2.shape)

    df2 = df2.with_columns((pl.col("Mixed DOT") - pl.col("mixed_mean_DOT")).alias("normalized_mixed_DOT"))

    df2 = df2.with_columns(pl.col("dominant_refill_trend").cast(pl.Utf8))

    df2 = df2.with_columns([
        pl.when(pl.col("dominant_refill_trend") == "Improving").then(1).otherwise(0).alias("improving_flag"),
        pl.when(pl.col("dominant_refill_trend") == "Declining").then(1).otherwise(0).alias("declining_flag"),
        pl.when(pl.col("dominant_refill_trend") == "Stable").then(1).otherwise(0).alias("stable_flag"),
    ])

    # Aggregate the counts for each trend flag per customer
    trend_counts = df2.group_by("cust_no").agg([
        pl.col("improving_flag").sum().alias("num_improving_products"),
        pl.col("declining_flag").sum().alias("num_declining_products"),
        pl.col("stable_flag").sum().alias("num_stable_products"),
    ])

    # Join these counts back to df2
    df2 = df2.join(trend_counts, on="cust_no", how="inner")
    # Group by customer and calculate the required features
    customer_features = df2.group_by("cust_no").agg([
        pl.col("num_improving_products").first(),
        pl.col("num_declining_products").first(),
        pl.col("num_stable_products").first(),
        pl.col("adherence_risk_score").mean().alias("avg_adherence_risk_score"),
    ])
    df2 = df2.with_columns(pl.col("dominant_refill_trend").cast(pl.Float32))
    # Join these features back to df2
    df2 = df2.join(customer_features, on="cust_no", how="inner")


    df2 = clean_company_names(df=df2, column_name='Product Company')
    df2 = clean_company_names(df=df2, column_name= 'Main Supplier')
    df2 = normalize_baseproducts(df=df2, column_name='Product Basename')
    df2 = normalize_baseproducts(df=df2, column_name='MolText')
    print(df2.shape)
    return df2

def prepare_data(df, columns_to_encode = [
    'title_txt', 'adr_typ', 'district_cde', 'sex_cde', 'language_cde', 
    'action_desc',  'Product Unit Container', 'Product Unit', 'Product Type',
    "orig_ord_type_cde", "ord_type_cde", "ord_line_type_id", "hp_cde", "shp_type_cde", 
    "cool_class_cde", "doc_type_cde", "line_discount_cde", "interact_cde", "pay_rspbl", 
    "pay_rspbl_type", "Product Availability", 'Product Dispension Type', 'Product Company','Product Class PCG','Main Supplier','MolText','Product Description',
    'Product Dispension Type Class', 'Product Generics Code', 'Product Basename', "ART_BASENAME", 
    "Product ATC (EPHMRA)", "Product Segment", 'iqvia_stop', 'pipeline_actual', 'pipeline_history', 
    'pipeline_start', 'pipeline_end', 'Product ATC (WHO)', 'ATC (WHO) level 1', 'ATC (WHO) level 2', 
    'ATC (WHO) level 3', 'ATC (WHO) level 4', 'ATC (WHO) level 5', 'Product Category Management', 
    'Category Management level 1', 'Category Management level 2', 'Category Management level 3', 
    'Category Management level 4', 'Category Management level 5', 'Product Therapeutic Index Code', 
    'MONO COMBI', 'prev_product_atc', 'CONCT', "ART_FORM", 'ART_FULLNAME']):
    vectorizer = ColumnVectorizer()
    df_transformed = vectorizer.fit_transform(df, label_cols=columns_to_encode)

    # only take numerical columns (numerical values )
    dict = {}
    for i,e in enumerate(df_transformed.columns):
        if df_transformed[e].dtype == pl.Float32 or df_transformed[e].dtype == pl.Int64: # only take numerical columns 
            dict[e] = i
    X  = df_transformed.select(list(dict.keys()))
    # only take 10000 rows
    X = X.head(10000)
    print(X.shape)

    # define label - target (acitve_flag) as customers being active / incactie in last 6 months
    # Step 1: Define Targets and Feature Engineering
    # Define active flag (target) for the past 6 months
    X = X.with_columns(
        pl.when(pl.col("Days Since Last Order") <= 180)
        .then(1)
        .otherwise(0)
        .alias("active_flag")
    )

    # Define inactive flag for customers inactive for over a year
    X = X.with_columns(
        pl.when(pl.col("Days Since Last Order") > 365)
        .then(1)
        .otherwise(0)
        .alias("inactive_flag")
    )

    # Drop the original target-related columns from the feature set
    df_features = X.drop(["active_flag", "inactive_flag", "Days Since Last Order"])

    # Combine the features and targets into a single training DataFrame
    train_df = X.select(["cust_no", "active_flag"]).join(df_features, on="cust_no", how="inner")
    return train_df


# Split data into train, validation, and test sets
def split_data(df):
    unique_customers = df["cust_no"].unique().to_numpy()
    np.random.seed(42)
    np.random.shuffle(unique_customers)

    num_customers = len(unique_customers)
    train_size = int(0.7 * num_customers)
    val_size = int(0.2 * num_customers)

    train_customers = unique_customers[:train_size]
    val_customers = unique_customers[train_size:train_size + val_size]
    test_customers = unique_customers[train_size + val_size:]

    train_df = df.filter(pl.col("cust_no").is_in(pl.Series(train_customers)))
    val_df = df.filter(pl.col("cust_no").is_in(pl.Series(val_customers)))
    test_df = df.filter(pl.col("cust_no").is_in(pl.Series(test_customers)))

    # Save splits
    train_df.write_csv(f"{PROCESSED_DATA_PATH}{config['TRAIN_FILE']}")
    val_df.write_csv(f"{PROCESSED_DATA_PATH}{config['VAL_FILE']}")
    test_df.write_csv(f"{PROCESSED_DATA_PATH}{config['TEST_FILE']}")
    return train_df, val_df, test_df

# Train XGBoost model
def train_model(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric="auc")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=0)
        y_val_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_val_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Train best model
    best_params = study.best_params
    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="auc")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
    return model

# Evaluate the model
def evaluate_model(model, X, y, set_name="Test"):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    print(f"{set_name} Set AUC: {auc:.4f}")
    print(classification_report(y, y_pred))
    return auc

# Explainability using SHAP
def explain_model(model, X, feature_names):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names)
    return shap_values

# Main pipeline
def main():
    # Step 1: Load data
    df, art_info, molecules = load_data()

    # Step 2: Preprocess data
    df = preprocess_data(df, art_info, molecules)

    # Step 3: Split data
    train_df, val_df, test_df = split_data(df)

    # Step 4: Prepare datasets
    feature_cols = [col for col in train_df.columns if col not in ["cust_no", TARGET_COLUMN]]
    X_train, y_train = train_df.select(feature_cols).to_numpy(), train_df[TARGET_COLUMN].to_numpy()
    X_val, y_val = val_df.select(feature_cols).to_numpy(), val_df[TARGET_COLUMN].to_numpy()
    X_test, y_test = test_df.select(feature_cols).to_numpy(), test_df[TARGET_COLUMN].to_numpy()

    # Step 5: Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # Step 6: Evaluate model
    evaluate_model(model, X_val, y_val, set_name="Validation")
    evaluate_model(model, X_test, y_test, set_name="Test")

    # Step 7: Explain model
    explain_model(model, X_test, feature_names=feature_cols)

if __name__ == "__main__":
    main()