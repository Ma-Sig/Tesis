"""
Data Loading and Preprocessing MCP Server

Provides tools for loading CSV/Excel files and basic preprocessing operations.
"""

import argparse
import os
import pandas as pd
import numpy as np
from typing import List, Any, Optional
from fastmcp import FastMCP
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataLoadingServer")

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

logger.info(f"Data Loading Server: Using data directory: {DATA_DIR}")

# Create FastMCP server instance
mcp = FastMCP("Data Loading Server")


def resolve_filepath(filepath: str) -> str:
    """Resolve filepath to absolute path.

    If filepath is absolute, use it as-is.
    If relative, resolve relative to DATA_DIR.
    """
    if os.path.isabs(filepath):
        return filepath
    else:
        return os.path.join(DATA_DIR, filepath)  # type: ignore


@mcp.tool()
def get_server_info(
    tool: Optional[str] = None,
    server: Optional[str] = None,
    arguments: Optional[Any] = None,
    filepath: Optional[str] = None,
) -> dict:
    """Get information about the data loading server configuration and available files.

    Returns:
        Dictionary with server info, data directory path, and list of available files
    """
    logger.info("🔍 get_server_info() called")
    try:
        available_files = []
        if os.path.exists(DATA_DIR):
            available_files = [
                {
                    "filename": f,
                    "size_bytes": os.path.getsize(os.path.join(DATA_DIR, f)),
                    "is_file": os.path.isfile(os.path.join(DATA_DIR, f)),
                }
                for f in os.listdir(DATA_DIR)
            ]

        result = {
            "success": True,
            "data_directory": DATA_DIR,
            "data_dir_exists": os.path.exists(DATA_DIR),
            "available_files": available_files,
            "file_count": len(available_files),
            "working_directory": os.getcwd(),
        }
        logger.info(f"✓ get_server_info() returning {len(available_files)} files")
        return result
    except Exception as e:
        logger.error(f"✗ get_server_info() error: {e}")
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "data_directory": DATA_DIR,
        }


@mcp.tool()
def load_csv(filepath: str, encoding: str = "utf-8", delimiter: str = ",") -> dict:
    """Load a CSV file and return metadata.

    Args:
        filepath: Path to CSV file (absolute or relative to data directory)
        encoding: File encoding (default: utf-8)
        delimiter: Column delimiter (default: comma)

    Returns:
        Dictionary with file metadata including rows, columns, dtypes, and sample data
    """
    original_filepath = filepath
    try:
        # Resolve path
        filepath = resolve_filepath(filepath)

        # Check if file exists
        if not os.path.exists(filepath):
            return {
                "success": False,
                "error": f"File not found: {filepath}",
                "data_directory": DATA_DIR,
                "requested_file": original_filepath,
            }

        # Load CSV
        df = pd.read_csv(filepath, encoding=encoding, delimiter=delimiter)

        return {
            "success": True,
            "filepath": filepath,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(5).to_dict(orient="records"),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "resolved_filepath": (
                filepath if "filepath" in locals() else original_filepath
            ),
            "data_directory": DATA_DIR,
        }


@mcp.tool()
def get_column_info(filepath: str, column_name: str):
    """Get detailed information about a specific column.

    Args:
        filepath: Path to CSV file (absolute or relative to data directory)
        column_name: Name of the column to analyze

    Returns:
        Dictionary with column statistics and info
    """
    try:
        filepath = resolve_filepath(filepath)

        if not os.path.exists(filepath):
            return {
                "error": f"File not found: {filepath}",
                "hint": f"Data directory: {DATA_DIR}",
            }

        df = pd.read_csv(filepath)

        if column_name not in df.columns:
            return {"error": f"Column '{column_name}' not found"}

        col = df[column_name]

        info = {
            "column_name": column_name,
            "dtype": str(col.dtype),
            "non_null_count": int(col.count()),
            "null_count": int(col.isna().sum()),
            "unique_count": int(col.nunique()),
        }

        # Add numeric statistics if applicable
        if pd.api.types.is_numeric_dtype(col):
            info.update(
                {
                    "mean": float(col.mean()) if not pd.isna(col.mean()) else None,
                    "median": (
                        float(col.median()) if not pd.isna(col.median()) else None
                    ),
                    "std": float(col.std()) if not pd.isna(col.std()) else None,
                    "min": float(col.min()) if not pd.isna(col.min()) else None,
                    "max": float(col.max()) if not pd.isna(col.max()) else None,
                }
            )
        else:
            # For categorical/text data
            info["most_common"] = col.value_counts().head(5).to_dict()

        return info
    except Exception as e:
        return {"error": str(e), "filepath": filepath}


@mcp.tool()
def detect_missing_values(filepath: str):
    """Detect missing values in the dataset.

    Args:
        filepath: Path to CSV file (absolute or relative to data directory)

    Returns:
        Dictionary with missing value statistics per column
    """
    try:
        filepath = resolve_filepath(filepath)

        if not os.path.exists(filepath):
            return {
                "error": f"File not found: {filepath}",
                "hint": f"Data directory: {DATA_DIR}",
            }

        df = pd.read_csv(filepath)

        missing_info = {}
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                missing_info[col] = {
                    "count": null_count,
                    "percentage": round((null_count / len(df)) * 100, 2),
                }

        return {
            "total_rows": len(df),
            "columns_with_missing": len(missing_info),
            "missing_data": missing_info,
            "complete_columns": [col for col in df.columns if col not in missing_info],
        }
    except Exception as e:
        return {"error": str(e), "filepath": filepath}


@mcp.tool()
def detect_duplicates(filepath: str, subset: Optional[List[str]] = None) -> dict:
    """Detect duplicate rows in the dataset.

    Args:
        filepath: Path to CSV file (absolute or relative to data directory)
        subset: List of column names to check for duplicates (None = all columns)

    Returns:
        Dictionary with duplicate statistics
    """
    try:
        filepath = resolve_filepath(filepath)

        if not os.path.exists(filepath):
            return {
                "error": f"File not found: {filepath}",
                "hint": f"Data directory: {DATA_DIR}",
            }

        df = pd.read_csv(filepath)

        duplicates = df.duplicated(subset=subset, keep=False)
        dup_count = int(duplicates.sum())

        return {
            "total_rows": len(df),
            "duplicate_rows": dup_count,
            "percentage": round((dup_count / len(df)) * 100, 2),
            "unique_rows": len(df) - dup_count,
            "checked_columns": subset if subset else "all",
        }
    except Exception as e:
        return {"error": str(e), "filepath": filepath}


@mcp.tool()
def infer_data_types(filepath: str) -> dict:
    """Automatically infer and suggest optimal data types for columns.

    Args:
        filepath: Path to CSV file (absolute or relative to data directory)

    Returns:
        Dictionary with current and suggested data types
    """
    try:
        filepath = resolve_filepath(filepath)

        if not os.path.exists(filepath):
            return {
                "error": f"File not found: {filepath}",
                "hint": f"Data directory: {DATA_DIR}",
            }

        df = pd.read_csv(filepath)

        type_info = {}
        for col in df.columns:
            current_type = str(df[col].dtype)

            # Try to infer better type
            suggested_type = current_type
            if current_type == "object":
                # Check if it could be numeric
                try:
                    pd.to_numeric(df[col], errors="raise")
                    suggested_type = "numeric (float/int)"
                except (ValueError, TypeError):
                    # Check if it could be datetime
                    try:
                        pd.to_datetime(df[col], errors="raise")
                        suggested_type = "datetime"
                    except (ValueError, TypeError):
                        # Check if it's categorical
                        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique
                            suggested_type = "categorical"

            type_info[col] = {
                "current": current_type,
                "suggested": suggested_type,
                "unique_values": int(df[col].nunique()),
                "unique_ratio": round(df[col].nunique() / len(df), 3),
            }

        return type_info
    except Exception as e:
        return {"error": str(e), "filepath": filepath}


@mcp.tool()
def get_basic_stats(filepath: str) -> dict:
    """Get basic statistical summary of the dataset.

    Args:
        filepath: Path to CSV file (absolute or relative to data directory)

    Returns:
        Dictionary with statistical summary
    """
    try:
        filepath = resolve_filepath(filepath)

        if not os.path.exists(filepath):
            return {
                "error": f"File not found: {filepath}",
                "hint": f"Data directory: {DATA_DIR}",
            }

        df = pd.read_csv(filepath)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        return {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "total_missing": int(df.isna().sum().sum()),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": (
                {
                    col: {
                        "unique": int(df[col].nunique()),
                        "top": df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                        "freq": (
                            int(df[col].value_counts().iloc[0])
                            if len(df[col]) > 0
                            else 0
                        ),
                    }
                    for col in categorical_cols
                }
                if categorical_cols
                else {}
            ),
        }
    except Exception as e:
        return {"error": str(e), "filepath": filepath}


@mcp.tool()
def read_file_head(filepath: str, n_rows: int = 10) -> dict:
    """Read first N rows of a CSV file.

    Args:
        filepath: Path to CSV file (absolute or relative to data directory)
        n_rows: Number of rows to read (default: 10)

    Returns:
        Dictionary with first N rows
    """
    original_filepath = filepath
    try:
        filepath = resolve_filepath(filepath)

        if not os.path.exists(filepath):
            # List files in data directory to help user
            available_files = []
            if os.path.exists(DATA_DIR):
                try:
                    available_files = os.listdir(DATA_DIR)
                except OSError:
                    pass

            return {
                "success": False,
                "error": f"File not found: {filepath}",
                "data_directory": DATA_DIR,
                "requested_file": original_filepath,
                "available_files": available_files,
                "suggestion": "Use just the filename if it's in data/uploads/, or provide the full absolute path",
            }

        df = pd.read_csv(filepath, nrows=n_rows)

        return {
            "success": True,
            "rows_read": len(df),
            "columns": df.columns.tolist(),
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "resolved_filepath": (
                filepath if "filepath" in locals() else original_filepath
            ),
            "data_directory": DATA_DIR,
        }


@mcp.tool()
def validate_csv_structure(
    filepath: str, expected_columns: Optional[List[str]] = None
) -> dict:
    """Validate CSV file structure and integrity.

    Args:
        filepath: Path to CSV file (absolute or relative to data directory)
        expected_columns: List of expected column names (optional)

    Returns:
        Dictionary with validation results
    """
    try:
        filepath = resolve_filepath(filepath)

        if not os.path.exists(filepath):
            return {
                "valid": False,
                "issues": [f"File not found: {filepath}"],
                "hint": f"Data directory: {DATA_DIR}",
            }

        df = pd.read_csv(filepath)

        issues = []

        # Check for empty file
        if len(df) == 0:
            issues.append("File is empty")

        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if "Unnamed" in str(col)]
        if unnamed_cols:
            issues.append(f"Found unnamed columns: {unnamed_cols}")

        # Check expected columns
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_columns)
            if missing_cols:
                issues.append(f"Missing expected columns: {list(missing_cols)}")
            if extra_cols:
                issues.append(f"Extra columns found: {list(extra_cols)}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
        }
    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Error reading file: {str(e)}"],
            "rows": 0,
            "columns": 0,
        }


# Run the server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loading MCP Server")
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
        "--port", type=int, default=8002, help="Port to bind to (for HTTP)"
    )
    args = parser.parse_args()

    if args.transport == "http":
        print(f"Starting Data Loading MCP Server on http://{args.host}:{args.port}/mcp")
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()
