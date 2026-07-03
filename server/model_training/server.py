"""
Model Training Tools MCP Server

Provides machine learning model training, hyperparameter tuning, and optimization tools.
"""

import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import pickle
from fastmcp import FastMCP
from pathlib import Path
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ModelTrainingServer")

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

logger.info(f"Model Training Server: Using data directory: {DATA_DIR}")

# Create FastMCP server instance
mcp = FastMCP("Model Training Server")


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
def train_classification_model(
    filepath: str,
    target_column: str,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    model_save_path: Optional[str] = None,
) -> dict:
    """Train a classification model.

    Args:
        filepath: Path to CSV file
        target_column: Name of the target column
        model_type: Type of model ('logistic', 'decision_tree', 'random_forest', 'svm', 'knn', 'naive_bayes', 'gradient_boosting')
        test_size: Proportion of test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        model_save_path: Optional path to save trained model

    Returns:
        Dictionary with training results and metrics
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical features (simple label encoding)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.factorize(X[col])[0]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Select and train model
    models = {
        "logistic": LogisticRegression(max_iter=1000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "svm": SVC(random_state=random_state),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    if model_type not in models:
        return {"error": f"Unknown model type: {model_type}"}

    model = models[model_type]
    model.fit(X_train, y_train)

    # Evaluate
    train_score = float(model.score(X_train, y_train))
    test_score = float(model.score(X_test, y_test))

    # Save model if path provided
    if model_save_path:
        model_save_path = resolve_filepath(model_save_path)
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)

    return {
        "model_type": model_type,
        "task": "classification",
        "train_accuracy": round(train_score, 4),
        "test_accuracy": round(test_score, 4),
        "overfitting_gap": round(train_score - test_score, 4),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X.columns.tolist(),
        "n_classes": int(len(np.unique(y))),
        "model_saved": model_save_path if model_save_path else None,
    }


@mcp.tool()
def train_regression_model(
    filepath: str,
    target_column: str,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    model_save_path: Optional[str] = None,
) -> dict:
    """Train a regression model.

    Args:
        filepath: Path to CSV file
        target_column: Name of the target column
        model_type: Type of model ('linear', 'ridge', 'lasso', 'decision_tree', 'random_forest', 'svr', 'knn', 'gradient_boosting')
        test_size: Proportion of test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        model_save_path: Optional path to save trained model

    Returns:
        Dictionary with training results and metrics
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.factorize(X[col])[0]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Select and train model
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(random_state=random_state),
        "lasso": Lasso(random_state=random_state),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state
        ),
        "svr": SVR(),
        "knn": KNeighborsRegressor(),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }

    if model_type not in models:
        return {"error": f"Unknown model type: {model_type}"}

    model = models[model_type]
    model.fit(X_train, y_train)

    # Evaluate
    train_r2 = float(model.score(X_train, y_train))
    test_r2 = float(model.score(X_test, y_test))

    # Calculate additional metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))

    # Save model if path provided
    if model_save_path:
        model_save_path = resolve_filepath(model_save_path)
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)

    return {
        "model_type": model_type,
        "task": "regression",
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "overfitting_gap": round(train_r2 - test_r2, 4),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X.columns.tolist(),
        "model_saved": model_save_path if model_save_path else None,
    }


@mcp.tool()
def cross_validate_model(
    filepath: str,
    target_column: str,
    model_type: str = "random_forest",
    task: str = "classification",
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Perform cross-validation on a model.

    Args:
        filepath: Path to CSV file
        target_column: Name of the target column
        model_type: Type of model
        task: 'classification' or 'regression'
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with cross-validation scores
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.factorize(X[col])[0]

    # Select model
    if task == "classification":
        models = {
            "logistic": LogisticRegression(max_iter=1000, random_state=random_state),
            "decision_tree": DecisionTreeClassifier(random_state=random_state),
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=random_state
            ),
            "svm": SVC(random_state=random_state),
            "knn": KNeighborsClassifier(),
            "naive_bayes": GaussianNB(),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        }
    else:
        models = {
            "linear": LinearRegression(),
            "ridge": Ridge(random_state=random_state),
            "lasso": Lasso(random_state=random_state),
            "decision_tree": DecisionTreeRegressor(random_state=random_state),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=random_state
            ),
            "svr": SVR(),
            "knn": KNeighborsRegressor(),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        }

    if model_type not in models:
        return {"error": f"Unknown model type: {model_type}"}

    model = models[model_type]

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv_folds)

    return {
        "model_type": model_type,
        "task": task,
        "cv_folds": cv_folds,
        "scores": [round(float(s), 4) for s in scores],
        "mean_score": round(float(scores.mean()), 4),
        "std_score": round(float(scores.std()), 4),
        "min_score": round(float(scores.min()), 4),
        "max_score": round(float(scores.max()), 4),
    }


@mcp.tool()
def hyperparameter_tuning(
    filepath: str,
    target_column: str,
    model_type: str = "random_forest",
    task: str = "classification",
    param_grid: Optional[Dict[str, List]] = None,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict:
    """Perform hyperparameter tuning using GridSearchCV.

    Args:
        filepath: Path to CSV file
        target_column: Name of the target column
        model_type: Type of model ('random_forest', 'decision_tree', 'svm')
        task: 'classification' or 'regression'
        param_grid: Dictionary of parameters to search (None = default grid)
        cv_folds: Number of cross-validation folds
        random_state: Random seed

    Returns:
        Dictionary with best parameters and scores
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.factorize(X[col])[0]

    # Default parameter grids
    default_grids = {
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "decision_tree": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "svm": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    }

    if param_grid is None:
        param_grid = default_grids.get(model_type, {})

    # Select model
    if task == "classification":
        models = {
            "random_forest": RandomForestClassifier(random_state=random_state),
            "decision_tree": DecisionTreeClassifier(random_state=random_state),
            "svm": SVC(random_state=random_state),
        }
    else:
        models = {
            "random_forest": RandomForestRegressor(random_state=random_state),
            "decision_tree": DecisionTreeRegressor(random_state=random_state),
            "svm": SVR(),
        }

    if model_type not in models:
        return {"error": f"Model type '{model_type}' not supported for tuning"}

    # Perform grid search
    grid_search = GridSearchCV(models[model_type], param_grid, cv=cv_folds, n_jobs=-1)
    grid_search.fit(X, y)

    return {
        "model_type": model_type,
        "task": task,
        "best_params": grid_search.best_params_,
        "best_score": round(float(grid_search.best_score_), 4),
        "cv_folds": cv_folds,
        "n_combinations_tested": len(grid_search.cv_results_["params"]),
    }


@mcp.tool()
def get_feature_importance(
    filepath: str,
    target_column: str,
    model_type: str = "random_forest",
    task: str = "classification",
    top_n: int = 10,
) -> dict:
    """Get feature importance from tree-based models.

    Args:
        filepath: Path to CSV file
        target_column: Name of the target column
        model_type: Type of model ('random_forest', 'decision_tree', 'gradient_boosting')
        task: 'classification' or 'regression'
        top_n: Number of top features to return

    Returns:
        Dictionary with feature importance scores
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    feature_names = X.columns.tolist()

    # Handle categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.factorize(X[col])[0]

    # Select model
    if task == "classification":
        models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
        }
    else:
        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
        }

    if model_type not in models:
        return {
            "error": f"Model type '{model_type}' doesn't support feature importance"
        }

    # Train model
    model = models[model_type]
    model.fit(X, y)

    # Get feature importance
    importances = model.feature_importances_

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [
        {
            "feature": feature_names[i],
            "importance": round(float(importances[i]), 4),
            "rank": rank + 1,
        }
        for rank, i in enumerate(indices)
    ]

    return {
        "model_type": model_type,
        "task": task,
        "top_features": top_features,
        "total_features": len(feature_names),
    }


@mcp.tool()
def compare_models(
    filepath: str,
    target_column: str,
    task: str = "classification",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Compare multiple models on the same dataset.

    Args:
        filepath: Path to CSV file
        target_column: Name of the target column
        task: 'classification' or 'regression'
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        Dictionary comparing all available models
    """
    filepath = resolve_filepath(filepath)

    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.factorize(X[col])[0]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Select models
    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=random_state
            ),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=random_state),
            "Lasso": Lasso(random_state=random_state),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=random_state
            ),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
            "KNN": KNeighborsRegressor(),
        }

    # Train and evaluate all models
    results = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            train_score = float(model.score(X_train, y_train))
            test_score = float(model.score(X_test, y_test))

            results.append(
                {
                    "model": name,
                    "train_score": round(train_score, 4),
                    "test_score": round(test_score, 4),
                    "overfitting": round(train_score - test_score, 4),
                }
            )
        except Exception as e:
            results.append({"model": name, "error": str(e)})

    # Sort by test score
    results.sort(key=lambda x: x.get("test_score", 0), reverse=True)

    return {
        "task": task,
        "models_compared": len(results),
        "results": results,
        "best_model": results[0]["model"] if results else None,
    }


# Run the server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training MCP Server")
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
        "--port", type=int, default=8004, help="Port to bind to (for HTTP)"
    )
    args = parser.parse_args()

    if args.transport == "http":
        print(
            f"Starting Model Training MCP Server on http://{args.host}:{args.port}/mcp"
        )
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()
