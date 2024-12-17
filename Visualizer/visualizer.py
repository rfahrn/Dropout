import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any

class Visualizer:
    """
    A reusable Visualizer class to create visual summaries for transaction-level data.
    
    Parameters
    ----------
    cust_column : str, optional
        Column name for customer ID (default='cust_no').
    date_column : str, optional
        Column name for transaction date (default='transaction_dte').
    style : str, optional
        Matplotlib style for visuals.
    """

    def __init__(
        self,cust_column: str = "cust_no",date_column: str = "transaction_dte",style: str = "seaborn-v0_8-darkgrid",):
        self.cust_column = cust_column
        self.date_column = date_column
        plt.style.use(style)
        sns.set_context("talk")

    def filter_customers(self, df: pl.DataFrame, cust_list: Optional[List[Any]] = None) -> pl.DataFrame:
        """Filter the data to include only specific customers."""
        if cust_list:
            df = df.filter(pl.col(self.cust_column).is_in(cust_list))
        return df

    def plot_customer_orders(self, df: pl.DataFrame, column: str = "order_frequency", cust_list: Optional[List[Any]] = None):
        """Plot the distribution of order frequency per customer."""
        df = self.filter_customers(df, cust_list)
        data = df.select([column]).to_pandas()
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], bins=30, kde=True, color="steelblue")
        plt.title("Distribution of Customer Orders")
        plt.xlabel("Order Frequency")
        plt.ylabel("Count")
        plt.show()

    def plot_adherence_scores(self, df: pl.DataFrame, column: str = "adherence_score", cust_list: Optional[List[Any]] = None):
        """Visualize adherence scores."""
        df = self.filter_customers(df, cust_list)
        data = df.select([column]).to_pandas()
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True, color="green")
        plt.title("Adherence Scores Distribution")
        plt.xlabel("Adherence Score")
        plt.ylabel("Density")
        plt.show()

    def plot_time_trends(self, df: pl.DataFrame, cust_list: Optional[List[Any]] = None):
        """Plot the transaction trend over time for specific customers."""
        df = self.filter_customers(df, cust_list)
        data = df.select([self.date_column]).to_pandas()
        data[self.date_column] = pd.to_datetime(data[self.date_column])
        data = data.groupby(data[self.date_column].dt.to_period("M")).size()
        plt.figure(figsize=(12, 6))
        data.plot(kind="line", color="royalblue")
        plt.title("Transaction Trends Over Time")
        plt.xlabel("Time (Monthly)")
        plt.ylabel("Number of Transactions")
        plt.show()

    def plot_categorical_distribution(self, df: pl.DataFrame, column: str, cust_list: Optional[List[Any]] = None):
        """Plot the distribution of a categorical column."""
        df = self.filter_customers(df, cust_list)
        data = df.select([column]).to_pandas()
        plt.figure(figsize=(10, 6))
        sns.countplot(y=column, data=data, palette="coolwarm", order=data[column].value_counts().index)
        plt.title(f"Distribution of {column}")
        plt.xlabel("Count")
        plt.ylabel(column)
        plt.show()

    def plot_boxplot_for_numerical(self, df: pl.DataFrame, numerical_columns: List[str], cust_list: Optional[List[Any]] = None):
        """Create boxplots for numerical columns to check outliers."""
        df = self.filter_customers(df, cust_list)
        data = df.select(numerical_columns).to_pandas()
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data)
        plt.title("Boxplot for Numerical Columns")
        plt.xticks(rotation=45)
        plt.show()

    def compare_adherence_trends(self, df: pl.DataFrame, trend_column: str = "dominant_refill_trend", cust_list: Optional[List[Any]] = None):
        """Compare adherence trends (Improving, Stable, Declining)."""
        df = self.filter_customers(df, cust_list)
        data = df.select([trend_column]).to_pandas()
        plt.figure(figsize=(8, 6))
        sns.countplot(x=trend_column, data=data, palette="muted")
        plt.title("Adherence Trends")
        plt.xlabel("Trend")
        plt.ylabel("Count")
        plt.show()

# Example Usage
if __name__ == "__main__":
    import polars as pl
    import pandas as pd

    # Sample Polars DataFrame for testing
    df = pl.DataFrame({
        "cust_no": [1, 2, 1, 3, 2, 3, 1, 2, 3],
        "order_frequency": [5, 3, 8, 2, 6, 4, 7, 2, 5],
        "adherence_score": [0.9, 0.7, 0.5, 0.6, 0.85, 0.92, 0.78, 0.3, 0.65],
        "dominant_refill_trend": ["Improving", "Stable", "Declining", "Stable", "Improving", "Declining", "Stable", "Improving", "Stable"],
        "transaction_dte": ["2023-01-01", "2023-01-15", "2023-02-01", "2023-03-01", "2023-04-01", "2023-04-15", "2023-05-01", "2023-05-15", "2023-06-01"]
    })

    visualizer = Visualizer()

    # Example: Focus on specific customers
    cust_list = [1, 2]

    # Plot customer order distribution
    visualizer.plot_customer_orders(df, column="order_frequency", cust_list=cust_list)

    # Plot adherence scores for selected customers
    visualizer.plot_adherence_scores(df, column="adherence_score", cust_list=cust_list)

    # Plot time trends
    visualizer.plot_time_trends(df, cust_list=cust_list)

    # Compare adherence trends
    visualizer.compare_adherence_trends(df, trend_column="dominant_refill_trend", cust_list=cust_list)
