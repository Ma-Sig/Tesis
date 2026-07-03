"""
Data Preparation Tools MCP Server

Provides advanced data transformation, feature engineering, and preparation tools.
"""

import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from fastmcp import FastMCP
from pathlib import Path
import json
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataPreparationServer")

# Configure data directory
DATA_DIR_ENV = os.getenv("MCP_DATA_DIR")
if DATA_DIR_ENV is None:
    # Use local data directory relative to this script
    SCRIPT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = str(SCRIPT_DIR / "data" / "uploads")
    # Create directory if it doesn't exist
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
else:
    DATA_DIR = DATA_DIR_ENV

logger.info(f"Data Preparation Server: Using data directory: {DATA_DIR}")

# Create FastMCP server instance
mcp = FastMCP("Data Preparation Server")


def resolve_filepath(filepath: str) -> str:
    """Resolve filepath to absolute path.

    If filepath is absolute, use it as-is.
    If relative, resolve relative to DATA_DIR.
    """
    if os.path.isabs(filepath):
        return filepath
    else:
        return os.path.join(DATA_DIR, filepath)


@mcp.tool()
def handle_missing_values(
    filepath: str,
    strategy: str = "mean",
    columns: Optional[List[str]] = None,
    output_filepath: Optional[str] = None,
) -> dict:
    """Handle missing values using various strategies.

    Args:
        filepath: Path to CSV file
        strategy: Imputation strategy ('mean', 'median', 'mode', 'constant', 'drop')
        columns: Specific columns to handle (None = all columns with missing values)
        output_filepath: Optional path to save cleaned data

    Returns:
        Dictionary with operation results
    """
    if not filepath.startswith("/app/"):
        filepath = f"/data/uploads/{filepath}"

    df = pd.read_csv(filepath)
    original_rows = len(df)

    if strategy == "drop":
        df = df.dropna(subset=columns) if columns else df.dropna()
        rows_removed = original_rows - len(df)
        result = {
            "strategy": "drop",
            "rows_removed": rows_removed,
            "remaining_rows": len(df),
        }
    else:
        # Select columns with missing values
        if columns is None:
            columns = df.columns[df.isna().any()].tolist()

        filled_columns = []
        for col in columns:
            if df[col].isna().any():
                if strategy in ["mean", "median"] and pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    if strategy == "mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].median(), inplace=True)
                    filled_columns.append(col)
                elif strategy == "mode":
                    df[col].fillna(
                        df[col].mode()[0] if len(df[col].mode()) > 0 else 0,
                        inplace=True,
                    )
                    filled_columns.append(col)
                elif strategy == "constant":
                    df[col].fillna(0, inplace=True)
                    filled_columns.append(col)

        result = {
            "strategy": strategy,
            "columns_filled": filled_columns,
            "total_filled": len(filled_columns),
        }

    # Save if output path provided
    if output_filepath:
        output_filepath = resolve_filepath(output_filepath)
        df.to_csv(output_filepath, index=False)
        result["saved_to"] = output_filepath

    result["remaining_missing"] = int(df.isna().sum().sum())
    return result


@mcp.tool()
def remove_duplicates(
    filepath: str,
    subset: Optional[List[str]] = None,
    keep: str = "first",
    output_filepath: Optional[str] = None,
) -> dict:
    """Remove duplicate rows from dataset.

    Args:
        filepath: Path to CSV file
        subset: Columns to consider for duplicates (None = all columns)
        keep: Which duplicates to keep ('first', 'last', False=remove all)
        output_filepath: Optional path to save cleaned data

    Returns:
        Dictionary with operation results
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)
    original_rows = len(df)

    # Convert "none" to False for pandas drop_duplicates
    if keep == "none":
        df_cleaned = df.drop_duplicates(subset=subset, keep=False)
    else:
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)  # type: ignore[arg-type]

    removed = original_rows - len(df_cleaned)

    if output_filepath:
        output_filepath = resolve_filepath(output_filepath)
        df_cleaned.to_csv(output_filepath, index=False)

    return {
        "original_rows": original_rows,
        "duplicates_removed": removed,
        "remaining_rows": len(df_cleaned),
        "saved_to": output_filepath if output_filepath else None,
    }


@mcp.tool()
def scale_features(
    filepath: str,
    columns: List[str],
    method: str = "standard",
    output_filepath: Optional[str] = None,
) -> dict:
    """Scale numerical features using standardization or normalization.

    Args:
        filepath: Path to CSV file
        columns: List of column names to scale
        method: Scaling method ('standard' or 'minmax')
        output_filepath: Optional path to save scaled data

    Returns:
        Dictionary with scaling results
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    # Verify columns exist and are numeric
    valid_columns = []
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            valid_columns.append(col)

    if not valid_columns:
        return {"error": "No valid numeric columns found"}

    # Scale
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df[valid_columns] = scaler.fit_transform(df[valid_columns])

    if output_filepath:
        output_filepath = resolve_filepath(output_filepath)
        df.to_csv(output_filepath, index=False)

    return {
        "method": method,
        "columns_scaled": valid_columns,
        "saved_to": output_filepath if output_filepath else None,
    }


@mcp.tool()
def encode_categorical(
    filepath: str,
    columns: List[str],
    method: str = "onehot",
    output_filepath: Optional[str] = None,
) -> dict:
    """Encode categorical variables.

    Args:
        filepath: Path to CSV file
        columns: List of column names to encode
        method: Encoding method ('onehot' or 'label')
        output_filepath: Optional path to save encoded data

    Returns:
        Dictionary with encoding results
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    encoded_columns = []

    for col in columns:
        if col not in df.columns:
            continue

        if method == "label":
            # Label encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoded_columns.append(col)
        elif method == "onehot":
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
            encoded_columns.append(col)
            encoded_columns.extend(dummies.columns.tolist())

    if output_filepath:
        output_filepath = resolve_filepath(output_filepath)
        df.to_csv(output_filepath, index=False)

    return {
        "method": method,
        "original_columns": columns,
        "result_columns": encoded_columns,
        "new_shape": {"rows": len(df), "columns": len(df.columns)},
        "saved_to": output_filepath if output_filepath else None,
    }


@mcp.tool()
def detect_outliers(
    filepath: str, column: str, method: str = "iqr", threshold: float = 1.5
) -> dict:
    """Detect outliers in a numerical column.

    Args:
        filepath: Path to CSV file
        column: Column name to check for outliers
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (IQR multiplier or z-score)

    Returns:
        Dictionary with outlier information
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric"}

    col_data = df[column].dropna()

    if method == "iqr":
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    else:  # zscore
        mean = col_data.mean()
        std = col_data.std()
        z_scores = np.abs((df[column] - mean) / std)
        outliers = df[z_scores > threshold]

    return {
        "column": column,
        "method": method,
        "threshold": threshold,
        "outlier_count": len(outliers),
        "outlier_percentage": round((len(outliers) / len(df)) * 100, 2),
        "outlier_indices": outliers.index.tolist()[:100],  # Limit to first 100
        "bounds": {
            "lower": float(lower_bound) if method == "iqr" else None,
            "upper": float(upper_bound) if method == "iqr" else None,
        },
    }


@mcp.tool()
def create_feature_bins(
    filepath: str,
    column: str,
    bins: int = 5,
    labels: Optional[List[str]] = None,
    output_filepath: Optional[str] = None,
) -> dict:
    """Create bins/categories from a continuous numerical column.

    Args:
        filepath: Path to CSV file
        column: Column name to bin
        bins: Number of bins to create
        labels: Optional labels for bins
        output_filepath: Optional path to save binned data

    Returns:
        Dictionary with binning results
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    new_col_name = f"{column}_binned"

    try:
        df[new_col_name] = pd.cut(df[column], bins=bins, labels=labels)

        if output_filepath:
            output_filepath = resolve_filepath(output_filepath)
            df.to_csv(output_filepath, index=False)

        return {
            "original_column": column,
            "new_column": new_col_name,
            "bins": bins,
            "value_counts": df[new_col_name].value_counts().to_dict(),
            "saved_to": output_filepath if output_filepath else None,
        }
    except Exception as e:
        return {"error": f"Binning failed: {str(e)}"}


@mcp.tool()
def split_dataset(
    filepath: str,
    train_ratio: float = 0.8,
    random_state: int = 42,
    shuffle: bool = True,
) -> dict:
    """Split dataset into training and testing sets.

    Args:
        filepath: Path to CSV file
        train_ratio: Ratio of training data (0.0 to 1.0)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle before splitting

    Returns:
        Dictionary with split information and saved file paths
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    # Generate output filenames
    base_name = filepath.rsplit("/", 1)[-1].replace(".csv", "")
    train_path = os.path.join(DATA_DIR, f"{base_name}_train.csv")
    test_path = os.path.join(DATA_DIR, f"{base_name}_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "original_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_ratio": train_ratio,
        "train_path": train_path,
        "test_path": test_path,
    }


@mcp.tool()
def filter_rows(
    filepath: str,
    column: str,
    condition: str,
    value: Any,
    output_filepath: Optional[str] = None,
) -> dict:
    """Filter rows based on a condition.

    Args:
        filepath: Path to CSV file
        column: Column name to filter on
        condition: Condition operator ('eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'contains')
        value: Value to compare against
        output_filepath: Optional path to save filtered data

    Returns:
        Dictionary with filtering results
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    original_rows = len(df)

    # Apply condition
    if condition == "eq":
        df_filtered = df[df[column] == value]
    elif condition == "ne":
        df_filtered = df[df[column] != value]
    elif condition == "gt":
        df_filtered = df[df[column] > value]
    elif condition == "lt":
        df_filtered = df[df[column] < value]
    elif condition == "gte":
        df_filtered = df[df[column] >= value]
    elif condition == "lte":
        df_filtered = df[df[column] <= value]
    elif condition == "contains":
        df_filtered = df[df[column].astype(str).str.contains(str(value), na=False)]
    else:
        return {"error": f"Unknown condition: {condition}"}

    if output_filepath:
        output_filepath = resolve_filepath(output_filepath)
        df_filtered.to_csv(output_filepath, index=False)

    return {
        "original_rows": original_rows,
        "filtered_rows": len(df_filtered),
        "rows_removed": original_rows - len(df_filtered),
        "condition": f"{column} {condition} {value}",
        "saved_to": output_filepath if output_filepath else None,
    }


@mcp.tool()
def rename_columns(
    filepath: str,
    column_mapping: Dict[str, str],
    output_filepath: Optional[str] = None,
) -> dict:
    """Rename one or more columns in the dataset.

    Args:
        filepath: Path to CSV file
        column_mapping: Dictionary mapping old column names to new names (e.g., {"old_name": "new_name"})
        output_filepath: Optional path to save renamed data

    Returns:
        Dictionary with renaming results
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    # Check which columns exist
    existing_columns = []
    missing_columns = []

    for old_name in column_mapping.keys():
        if old_name in df.columns:
            existing_columns.append(old_name)
        else:
            missing_columns.append(old_name)

    if not existing_columns:
        return {
            "error": "None of the specified columns exist in the dataset",
            "missing_columns": missing_columns,
            "available_columns": df.columns.tolist(),
        }

    # Rename only existing columns
    rename_dict = {k: v for k, v in column_mapping.items() if k in existing_columns}
    df.rename(columns=rename_dict, inplace=True)

    if output_filepath:
        output_filepath = resolve_filepath(output_filepath)
        df.to_csv(output_filepath, index=False)

    return {
        "renamed_columns": rename_dict,
        "columns_renamed_count": len(existing_columns),
        "missing_columns": missing_columns if missing_columns else None,
        "new_column_names": df.columns.tolist(),
        "saved_to": output_filepath if output_filepath else None,
    }


@mcp.tool()
def add_column_headers(
    filepath: str,
    column_names: List[str],
    output_filepath: Optional[str] = None,
    has_header: bool = False,
) -> dict:
    """Add or replace column headers in a dataset.

    Args:
        filepath: Path to CSV file
        column_names: List of new column names (must match number of columns)
        output_filepath: Optional path to save data with new headers
        has_header: Whether the file already has a header row (True = replace header, False = file has no header)

    Returns:
        Dictionary with operation results
    """
    filepath = resolve_filepath(filepath)

    # Read file with or without header
    if has_header:
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, header=None)

    # Validate column count
    if len(column_names) != len(df.columns):
        return {
            "error": f"Number of column names ({len(column_names)}) does not match number of columns in file ({len(df.columns)})",
            "expected_count": len(df.columns),
            "provided_count": len(column_names),
            "current_columns": df.columns.tolist(),
        }

    # Check for duplicate column names
    if len(column_names) != len(set(column_names)):
        duplicates = [name for name in column_names if column_names.count(name) > 1]
        return {
            "error": "Duplicate column names are not allowed",
            "duplicate_names": list(set(duplicates)),
        }

    # Assign new column names
    old_columns = df.columns.tolist()
    df.columns = column_names

    if output_filepath:
        output_filepath = resolve_filepath(output_filepath)
        df.to_csv(output_filepath, index=False)

    return {
        "previous_columns": old_columns,
        "new_columns": column_names,
        "columns_count": len(column_names),
        "rows_count": len(df),
        "saved_to": output_filepath if output_filepath else None,
    }


# Run the server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (stdio for local, http for network)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (for HTTP)"
    )
    parser.add_argument(
        "--port", type=int, default=8003, help="Port to bind to (for HTTP)"
    )
    args = parser.parse_args()

    if args.transport == "http":
        print(
            f"Starting Data Preparation MCP Server on http://{args.host}:{args.port}/mcp"
        )
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()
