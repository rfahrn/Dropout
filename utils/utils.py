
import polars as pl
from sklearn.preprocessing import LabelEncoder

def clean_company_names(df, column_name):
    suffix_pattern = r"\b(?:ag|gmbh|sa|ltd|inc|co|corporation|llc|sarl|bv|kg|se|nv|plc|limited)\b"
    df = df.with_columns(
        pl.col(column_name)
        .fill_null("Unknown")
        .str.strip_chars(" ") 
        .str.replace(r"(?i)\bna\b", "Unknown")  
        .str.to_lowercase()  
        .str.replace_all(r"[^\w\s&]", "") 
        .str.replace_all(r"\s+", " ")  
        .str.replace_all(r"&", "and") 
        .str.replace_all(suffix_pattern, "") 
        .str.strip_chars(" ") 
        .str.replace_all(r"\s", "_") 
        .alias(column_name))
    return df

def normalize_baseproducts(df, column_name):
    df = df.with_columns(pl.col(column_name).fill_null("Unknown").str.strip_chars("_").str.to_lowercase().str.replace_all(r"[ -+\.]", "_").str.replace(r"_{2,}", "_").str.strip_chars("_").alias(column_name))
    return df


class ColumnVectorizer:
    def __init__(self):
        self.encoders = {}

    def fit_label_encoder(self, df, columns):
        for col in columns:
            le = LabelEncoder()
            df = df.with_columns(pl.col(col).fill_null('Unknown').alias(col))
            le.fit(df[col].to_list())
            self.encoders[col] = le

    def transform_label_encoder(self, df, columns):
        for col in columns:
            if col in self.encoders:
                le = self.encoders[col]
                df = df.with_columns(pl.col(col).fill_null('Unknown').alias(col))
                transformed_values = le.transform(df[col].to_list())
                df = df.with_columns(pl.Series(col, transformed_values))
        return df

    def fit_transform(self, df, label_cols=None):
        if label_cols:
            self.fit_label_encoder(df, label_cols)
            df = self.transform_label_encoder(df, label_cols)
        return df

    def transform(self, df, label_cols=None):
        if label_cols:
            df = self.transform_label_encoder(df, label_cols)
        return df
