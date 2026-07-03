"""
Model Evaluation Tools MCP Server

Provides comprehensive model evaluation, metrics calculation, and visualization tools.
"""

import argparse
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
import pickle
import json
from fastmcp import FastMCP
from pathlib import Path
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ModelEvaluationServer")

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

logger.info(f"Model Evaluation Server: Using data directory: {DATA_DIR}")

# Registry file that accumulates evaluation results over time
RESULTS_REGISTRY = Path(DATA_DIR) / "evaluation_results.json"

# Create FastMCP server instance
mcp = FastMCP("Model Evaluation Server")


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
def evaluate_classification_model(
    model_path: str, test_data_path: str, target_column: str
) -> dict:
    """Evaluate a trained classification model.

    Args:
        model_path: Path to saved model pickle file
        test_data_path: Path to test dataset CSV
        target_column: Name of the target column

    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    # Load model
    model_path = resolve_filepath(model_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_data_path = resolve_filepath(test_data_path)

    df = pd.read_csv(test_data_path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Handle categorical features
    for col in X_test.select_dtypes(include=["object"]).columns:
        X_test[col] = pd.factorize(X_test[col])[0]

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = float(accuracy_score(y_test, y_pred))

    # For binary classification, calculate additional metrics
    is_binary = len(np.unique(y_test)) == 2

    if is_binary:
        precision = float(precision_score(y_test, y_pred, average="binary"))
        recall = float(recall_score(y_test, y_pred, average="binary"))
        f1 = float(f1_score(y_test, y_pred, average="binary"))
    else:
        precision = float(precision_score(y_test, y_pred, average="weighted"))
        recall = float(recall_score(y_test, y_pred, average="weighted"))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "test_samples": len(y_test),
        "n_classes": len(np.unique(y_test)),
        "is_binary": is_binary,
    }


@mcp.tool()
def evaluate_regression_model(
    model_path: str, test_data_path: str, target_column: str
) -> dict:
    """Evaluate a trained regression model.

    Args:
        model_path: Path to saved model pickle file
        test_data_path: Path to test dataset CSV
        target_column: Name of the target column

    Returns:
        Dictionary with regression evaluation metrics
    """
    # Load model
    model_path = resolve_filepath(model_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_data_path = resolve_filepath(test_data_path)

    df = pd.read_csv(test_data_path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Handle categorical features
    for col in X_test.select_dtypes(include=["object"]).columns:
        X_test[col] = pd.factorize(X_test[col])[0]

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = float(r2_score(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = (
        float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
        if (y_test != 0).all()
        else None
    )

    return {
        "r2_score": round(r2, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4) if mape else None,
        "test_samples": len(y_test),
        "prediction_range": {
            "min": round(float(y_pred.min()), 4),
            "max": round(float(y_pred.max()), 4),
        },
        "actual_range": {
            "min": round(float(y_test.min()), 4),
            "max": round(float(y_test.max()), 4),
        },
    }


@mcp.tool()
def get_classification_report(
    model_path: str, test_data_path: str, target_column: str
) -> dict:
    """Get detailed classification report.

    Args:
        model_path: Path to saved model pickle file
        test_data_path: Path to test dataset CSV
        target_column: Name of the target column

    Returns:
        Dictionary with per-class metrics
    """
    # Load model
    model_path = resolve_filepath(model_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_data_path = resolve_filepath(test_data_path)

    df = pd.read_csv(test_data_path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Handle categorical features
    for col in X_test.select_dtypes(include=["object"]).columns:
        X_test[col] = pd.factorize(X_test[col])[0]

    # Make predictions
    y_pred = model.predict(X_test)

    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "per_class_metrics": report,
        "classes": list(np.unique(y_test)),
        "overall_accuracy": round(float(report["accuracy"]), 4),  # type: ignore[index]
    }


@mcp.tool()
def calculate_confusion_matrix(
    model_path: str,
    test_data_path: str,
    target_column: str,
    normalize: Optional[str] = None,
) -> dict:
    """Calculate and optionally normalize confusion matrix.

    Args:
        model_path: Path to saved model pickle file
        test_data_path: Path to test dataset CSV
        target_column: Name of the target column
        normalize: Normalization mode ('true', 'pred', 'all', or None)

    Returns:
        Dictionary with confusion matrix
    """
    # Load model
    model_path = resolve_filepath(model_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_data_path = resolve_filepath(test_data_path)

    df = pd.read_csv(test_data_path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Handle categorical features
    for col in X_test.select_dtypes(include=["object"]).columns:
        X_test[col] = pd.factorize(X_test[col])[0]

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)  # type: ignore[arg-type]

    return {
        "confusion_matrix": cm.tolist(),
        "normalized": normalize if normalize else "none",
        "classes": list(np.unique(y_test)),
    }


@mcp.tool()
def predict_with_model(
    model_path: str, input_data_path: str, output_path: Optional[str] = None
) -> dict:
    """Make predictions using a trained model.

    Args:
        model_path: Path to saved model pickle file
        input_data_path: Path to input data CSV (without target column)
        output_path: Optional path to save predictions

    Returns:
        Dictionary with predictions
    """
    # Load model
    model_path = resolve_filepath(model_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load input data
    input_data_path = resolve_filepath(input_data_path)

    df = pd.read_csv(input_data_path)

    # Handle categorical features
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.factorize(df[col])[0]

    # Make predictions
    predictions = model.predict(df)

    # Add predictions to dataframe
    df["prediction"] = predictions

    # Save if output path provided
    if output_path:
        output_path = resolve_filepath(output_path)
        df.to_csv(output_path, index=False)

    return {
        "predictions": predictions.tolist()[:100],  # Limit to first 100
        "total_predictions": len(predictions),
        "unique_predictions": len(np.unique(predictions)),
        "saved_to": output_path if output_path else None,
    }


@mcp.tool()
def calculate_residuals(
    model_path: str, test_data_path: str, target_column: str
) -> dict:
    """Calculate prediction residuals for regression models.

    Args:
        model_path: Path to saved model pickle file
        test_data_path: Path to test dataset CSV
        target_column: Name of the target column

    Returns:
        Dictionary with residual statistics
    """
    # Load model
    model_path = resolve_filepath(model_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_data_path = resolve_filepath(test_data_path)

    df = pd.read_csv(test_data_path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Handle categorical features
    for col in X_test.select_dtypes(include=["object"]).columns:
        X_test[col] = pd.factorize(X_test[col])[0]

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate residuals
    residuals = y_test - y_pred

    return {
        "mean_residual": round(float(residuals.mean()), 4),
        "std_residual": round(float(residuals.std()), 4),
        "min_residual": round(float(residuals.min()), 4),
        "max_residual": round(float(residuals.max()), 4),
        "median_residual": round(float(np.median(residuals)), 4),
        "residual_range": round(float(residuals.max() - residuals.min()), 4),
        "samples": len(residuals),
    }


@mcp.tool()
def compare_model_predictions(
    model1_path: str,
    model2_path: str,
    test_data_path: str,
    target_column: str,
    task: str = "classification",
) -> dict:
    """Compare predictions from two different models.

    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        test_data_path: Path to test dataset CSV
        target_column: Name of the target column
        task: 'classification' or 'regression'

    Returns:
        Dictionary comparing both models
    """
    # Load models
    model1_path = resolve_filepath(model1_path)
    model2_path = resolve_filepath(model2_path)

    with open(model1_path, "rb") as f:
        model1 = pickle.load(f)
    with open(model2_path, "rb") as f:
        model2 = pickle.load(f)

    # Load test data
    test_data_path = resolve_filepath(test_data_path)

    df = pd.read_csv(test_data_path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Handle categorical features
    for col in X_test.select_dtypes(include=["object"]).columns:
        X_test[col] = pd.factorize(X_test[col])[0]

    # Make predictions
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)

    # Calculate metrics
    if task == "classification":
        score1 = float(accuracy_score(y_test, y_pred1))
        score2 = float(accuracy_score(y_test, y_pred2))
        metric = "accuracy"
    else:
        score1 = float(r2_score(y_test, y_pred1))
        score2 = float(r2_score(y_test, y_pred2))
        metric = "r2_score"

    # Agreement between models
    agreement = float(np.mean(y_pred1 == y_pred2))

    # Determine which model is better
    if score2 > score1:
        better_model = "model2"
    elif score1 > score2:
        better_model = "model1"
    else:
        better_model = "tie"

    return {
        "task": task,
        "metric": metric,
        "model1_score": round(score1, 4),
        "model2_score": round(score2, 4),
        "difference": round(score2 - score1, 4),
        "better_model": better_model,
        "agreement_ratio": round(agreement, 4),
        "test_samples": len(y_test),
    }


@mcp.tool()
def error_analysis(
    model_path: str,
    test_data_path: str,
    target_column: str,
    task: str = "classification",
) -> dict:
    """Perform error analysis on model predictions.

    Args:
        model_path: Path to saved model pickle file
        test_data_path: Path to test dataset CSV
        target_column: Name of the target column
        task: 'classification' or 'regression'

    Returns:
        Dictionary with error analysis
    """
    # Load model
    model_path = resolve_filepath(model_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_data_path = resolve_filepath(test_data_path)

    df = pd.read_csv(test_data_path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Handle categorical features
    for col in X_test.select_dtypes(include=["object"]).columns:
        X_test[col] = pd.factorize(X_test[col])[0]

    # Make predictions
    y_pred = model.predict(X_test)

    if task == "classification":
        # Find misclassified samples
        errors = y_test != y_pred
        error_rate = float(errors.mean())

        return {
            "task": "classification",
            "total_samples": len(y_test),
            "errors": int(errors.sum()),
            "error_rate": round(error_rate, 4),
            "accuracy": round(1 - error_rate, 4),
            "error_indices": np.nonzero(errors)[0].tolist()[
                :50
            ],  # First 50 error indices
        }
    else:
        # Find samples with large errors
        residuals = np.abs(y_test - y_pred)
        threshold = np.percentile(residuals, 90)  # Top 10% errors
        large_errors = residuals > threshold

        return {
            "task": "regression",
            "total_samples": len(y_test),
            "large_errors": int(large_errors.sum()),
            "error_threshold": round(float(threshold), 4),
            "mean_absolute_error": round(float(residuals.mean()), 4),
            "max_error": round(float(residuals.max()), 4),
            "large_error_indices": np.nonzero(large_errors)[0].tolist()[:50],
        }


@mcp.tool()
def save_evaluation_results(
    metrics: Dict[str, Any],
    task_type: str,
    model_name: str,
    dataset_name: str,
    notes: Optional[str] = None,
    tags: Optional[List[str]] = None,
    pipeline: Optional[Dict[str, Any]] = None,
    registry_path: Optional[str] = None,
) -> dict:
    """Save evaluation metrics to a persistent JSON registry file.

    Use this after running evaluate_classification_model or
    evaluate_regression_model (or any other evaluation tool) to record
    the results so they can be compared and analysed later.

    The optional ``pipeline`` argument lets you attach the full pipeline
    definition that produced these results, including every agent step,
    its type, id, and configuration.  Recommended shape::

        {
            "pipeline_id":   "<unique id from sim ai or generated>",
            "pipeline_name": "iris_classification_pipeline",
            "steps": [
                {"order": 1, "agent_type": "data_loader",
                 "agent_id": "agt_abc", "config": {...}},
                {"order": 2, "agent_type": "data_cleaner",
                 "agent_id": "agt_def", "config": {...}},
                {"order": 3, "agent_type": "normalizer",
                 "agent_id": "agt_ghi", "config": {...}},
                {"order": 4, "agent_type": "model_trainer",
                 "agent_id": "agt_jkl", "config": {...}},
                {"order": 5, "agent_type": "model_evaluator",
                 "agent_id": "agt_mno", "config": {...}},
            ]
        }

    Any extra keys are stored as-is, so you can include whatever
    sim-ai-specific metadata is useful.

    Args:
        metrics: Dictionary of metrics returned by an evaluation tool.
        task_type: 'classification' or 'regression'.
        model_name: Human-readable name / identifier for the model
                    (e.g. 'random_forest_v1', 'iris_logreg').
        dataset_name: Name or path of the dataset used for evaluation.
        notes: Optional free-text notes about this run.
        tags: Optional list of string tags (e.g. ['baseline', 'tuned']).
        pipeline: Optional pipeline definition dict (see above).
        registry_path: Optional custom path for the registry file.
                       Defaults to <DATA_DIR>/evaluation_results.json.

    Returns:
        Dictionary with the saved entry and the path to the registry file.
    """
    # Determine registry file location
    if registry_path:
        reg_file = Path(resolve_filepath(registry_path))
    else:
        reg_file = RESULTS_REGISTRY

    # Load existing records
    if reg_file.exists():
        try:
            with open(reg_file, "r") as f:
                records = json.load(f)
            if not isinstance(records, list):
                records = []
        except (json.JSONDecodeError, OSError):
            records = []
    else:
        records = []

    # Normalise pipeline: ensure pipeline_id is always present
    pipeline_meta: Optional[Dict[str, Any]] = None
    if pipeline is not None:
        pipeline_meta = dict(pipeline)  # shallow copy
        if "pipeline_id" not in pipeline_meta:
            import uuid

            pipeline_meta["pipeline_id"] = str(uuid.uuid4())

    entry: Dict[str, Any] = {
        "id": len(records) + 1,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "task_type": task_type,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "metrics": metrics,
        "notes": notes,
        "tags": tags or [],
        "pipeline": pipeline_meta,
    }

    records.append(entry)

    reg_file.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_file, "w") as f:
        json.dump(records, f, indent=2, default=str)

    logger.info(
        f"Saved evaluation result #{entry['id']} for model '{model_name}' "
        f"to {reg_file}"
    )

    return {
        "saved": True,
        "entry_id": entry["id"],
        "registry_file": str(reg_file),
        "total_records": len(records),
        "entry": entry,
    }


@mcp.tool()
def list_evaluation_results(
    task_type: Optional[str] = None,
    model_name: Optional[str] = None,
    tag: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    last_n: Optional[int] = None,
    registry_path: Optional[str] = None,
) -> dict:
    """List and filter evaluation results from the registry.

    Args:
        task_type: Filter by task type ('classification' or 'regression').
                   Pass None to return both.
        model_name: Filter by exact model name. Pass None for all models.
        tag: Return only records that contain this tag.
        pipeline_id: Return only records produced by this specific pipeline.
        last_n: Return only the most recent N records.
        registry_path: Optional custom path for the registry file.
                       Defaults to <DATA_DIR>/evaluation_results.json.

    Returns:
        Dictionary with matching records and summary statistics.
    """
    if registry_path:
        reg_file = Path(resolve_filepath(registry_path))
    else:
        reg_file = RESULTS_REGISTRY

    if not reg_file.exists():
        return {
            "records": [],
            "total_in_registry": 0,
            "returned": 0,
            "registry_file": str(reg_file),
            "message": "Registry file not found. No results have been saved yet.",
        }

    try:
        with open(reg_file, "r") as f:
            records: List[Dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return {"error": f"Could not read registry file: {exc}"}

    total = len(records)

    if task_type:
        records = [r for r in records if r.get("task_type") == task_type]
    if model_name:
        records = [r for r in records if r.get("model_name") == model_name]
    if tag:
        records = [r for r in records if tag in r.get("tags", [])]
    if pipeline_id:
        records = [
            r
            for r in records
            if (r.get("pipeline") or {}).get("pipeline_id") == pipeline_id
        ]

    if last_n and last_n > 0:
        records = records[-last_n:]

    summary: Dict[str, Any] = {}
    for rec in records:
        mname = rec.get("model_name", "unknown")
        if mname not in summary:
            summary[mname] = {
                "task_type": rec.get("task_type"),
                "runs": 0,
                "latest_metrics": {},
                "latest_timestamp": "",
            }
        summary[mname]["runs"] += 1
        summary[mname]["latest_metrics"] = rec.get("metrics", {})
        summary[mname]["latest_timestamp"] = rec.get("timestamp", "")

    return {
        "total_in_registry": total,
        "returned": len(records),
        "registry_file": str(reg_file),
        "records": records,
        "model_summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation MCP Server")
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
        "--port", type=int, default=8005, help="Port to bind to (for HTTP)"
    )
    args = parser.parse_args()

    if args.transport == "http":
        print(
            f"Starting Model Evaluation MCP Server on http://{args.host}:{args.port}/mcp"
        )
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()
