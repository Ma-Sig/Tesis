"""
Feature Engineering MCP Server

Provides tools for feature selection, dimensionality reduction, and data augmentation.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from typing import List
from fastmcp import FastMCP
from pathlib import Path
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FeatureEngineeringServer")

# Configure data directory
DATA_DIR_ENV = os.getenv("MCP_DATA_DIR")
if DATA_DIR_ENV is None:
    SCRIPT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = str(SCRIPT_DIR / "data" / "uploads")
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
else:
    DATA_DIR = DATA_DIR_ENV

logger.info(f"Feature Engineering Server: Using data directory: {DATA_DIR}")

mcp = FastMCP("Feature Engineering Server")


def resolve_filepath(filepath: str) -> str:
    if os.path.isabs(filepath):
        return filepath
    else:
        return os.path.join(DATA_DIR, filepath)


@mcp.tool()
def feature_selection(
    filepath: str, target_column: str, method: str = "f_classif", k: int = 10
) -> dict:
    """Select top-k features based on statistical methods.

    Args:
        filepath: Path to the dataset
        target_column: Name of the target column
        method: Feature selection method ('f_classif' or 'mutual_info')
        k: Number of top features to select

    Returns:
        Dictionary with selected features
    """
    filepath = resolve_filepath(filepath)
    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if method == "f_classif":
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        return {"error": f"Unsupported method '{method}'"}

    selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()

    return {"selected_features": selected_features}


@mcp.tool()
def dimensionality_reduction(filepath: str, n_components: int = 2) -> dict:
    """Reduce dimensionality of the dataset using PCA.

    Args:
        filepath: Path to the dataset
        n_components: Number of principal components to keep

    Returns:
        Dictionary with reduced data
    """
    filepath = resolve_filepath(filepath)
    df = pd.read_csv(filepath)

    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(df)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "reduced_data": reduced_data.tolist(),
    }


@mcp.tool()
def generate_synthetic_data(filepath: str, target_column: str) -> dict:
    """Generate synthetic data using SMOTE.

    Args:
        filepath: Path to the dataset
        target_column: Name of the target column

    Returns:
        Dictionary with synthetic data
    """
    filepath = resolve_filepath(filepath)
    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    smote = SMOTE()
    resampled = smote.fit_resample(X, y)

    if len(resampled) != 2:
        return {"error": "Unexpected return value from SMOTE's fit_resample method"}

    x_resampled, y_resampled = resampled

    resampled_df = pd.DataFrame(x_resampled, columns=X.columns)
    resampled_df[target_column] = y_resampled

    output_path = os.path.join(DATA_DIR, "synthetic_data.csv")
    resampled_df.to_csv(output_path, index=False)

    return {"output_path": output_path, "rows_generated": len(resampled_df) - len(df)}


@mcp.tool()
def add_noise(
    filepath: str, columns: List[str], noise_level: float = 0.01, seed: int = 42
) -> dict:
    """Add random noise to numerical features.

    Args:
        filepath: Path to the dataset
        columns: List of columns to add noise to
        noise_level: Standard deviation of the noise
        seed: Random seed for reproducibility

    Returns:
        Dictionary with the path to the noisy dataset
    """
    filepath = resolve_filepath(filepath)
    df = pd.read_csv(filepath)

    rng = np.random.default_rng(seed)
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            noise = rng.normal(0, noise_level, size=len(df))
            df[column] += noise

    output_path = os.path.join(DATA_DIR, "noisy_data.csv")
    df.to_csv(output_path, index=False)

    return {"output_path": output_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering MCP Server")
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
        "--port", type=int, default=8006, help="Port to bind to (for HTTP)"
    )
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()
