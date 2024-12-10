import argparse
from config import Config
from data_loader import DataLoader
from preprocessor import Preprocessor
from train import train_model
from evaluate import evaluate_model, explain_model
from utils import ColumnVectorizer
import polars as pl

def main(args):
    config = Config(args.config).config
    data_loader = DataLoader(config)
    preprocessor = Preprocessor()

    df, art_info, molecules = data_loader.load_datasets(segment=args.segment)
    df = preprocessor.preprocess(df, art_info, molecules)

    vectorizer = ColumnVectorizer()
    df_transformed = vectorizer.fit_transform(df, label_cols=args.columns_to_encode)
    X = df_transformed.select([col for col in df_transformed.columns if df_transformed[col].dtype in [pl.Float32, pl.Int64]])
    X = X.head(10000)

    X = X.with_columns(
        pl.when(pl.col("Days Since Last Order") <= 180).then(1).otherwise(0).alias("active_flag"),
        pl.when(pl.col("Days Since Last Order") > 365).then(1).otherwise(0).alias("inactive_flag"))
    df_features = X.drop(["active_flag", "inactive_flag", "Days Since Last Order"])
    train_df = X.select(["cust_no", "active_flag"]).join(df_features, on="cust_no", how="inner")

    unique_customers = train_df["cust_no"].unique().to_numpy()
    np.random.seed(42)
    np.random.shuffle(unique_customers)
    train_size = int(0.7 * len(unique_customers))
    val_size = int(0.2 * len(unique_customers))
    train_customers = unique_customers[:train_size]
    val_customers = unique_customers[train_size:train_size + val_size]
    test_customers = unique_customers[train_size + val_size:]
    train_df = train_df.filter(pl.col("cust_no").is_in(pl.Series(train_customers)))
    val_df = train_df.filter(pl.col("cust_no").is_in(pl.Series(val_customers)))
    test_df = train_df.filter(pl.col("cust_no").is_in(pl.Series(test_customers)))

    feature_cols = [col for col in train_df.columns if col not in ["cust_no", config["TARGET_COLUMN"]]]
    X_train, y_train = train_df.select(feature_cols).to_numpy(), train_df[config["TARGET_COLUMN"]].to_numpy()
    X_val, y_val = val_df.select(feature_cols).to_numpy(), val_df[config["TARGET_COLUMN"]].to_numpy()
    X_test, y_test = test_df.select(feature_cols).to_numpy(), test_df[config["TARGET_COLUMN"]].to_numpy()

    model = train_model(X_train, y_train, X_val, y_val)
    evaluate_model(model, X_val, y_val, set_name="Validation")
    evaluate_model(model, X_test, y_test, set_name="Test")
    explain_model(model, X_test, feature_names=feature_cols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and train model for adherence prediction.")
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to the config file.')
    parser.add_argument('--segment', type=str, default="all", help='Segment to filter data.')
    parser.add_argument('--columns_to_encode', nargs='+', default=[
        'title_txt', 'adr_typ', 'district_cde', 'sex_cde', 'language_cde', 
        'action_desc', 'Product Unit Container', 'Product Unit', 'Product Type',
        "orig_ord_type_cde", "ord_type_cde", "ord_line_type_id", "hp_cde", "shp_type_cde", 
        "cool_class_cde", "doc_type_cde", "line_discount_cde", "interact_cde", "pay_rspbl", 
        "pay_rspbl_type", "Product Availability", 'Product Dispension Type', 'Product Company','Product Class PCG','Main Supplier','MolText','Product Description',
        'Product Dispension Type Class', 'Product Generics Code', 'Product Basename', "ART_BASENAME", 
        "Product ATC (EPHMRA)", "Product Segment", 'iqvia_stop', 'pipeline_actual', 'pipeline_history', 
        'pipeline_start', 'pipeline_end', 'Product ATC (WHO)', 'ATC (WHO) level 1', 'ATC (WHO) level 2', 
        'ATC (WHO) level 3', 'ATC (WHO) level 4', 'ATC (WHO) level 5', 'Product Category Management', 
        'Category Management level 1', 'Category Management level 2', 'Category Management level 3', 
        'Category Management level 4', 'Category Management level 5', 'Product Therapeutic Index Code', 
        'MONO COMBI', 'prev_product_atc', 'CONCT', "ART_FORM", 'ART_FULLNAME'
    ], help='Columns to encode.')
    
    args = parser.parse_args()
    main(args)