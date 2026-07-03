# Model Evaluation Tools MCP Server

Provides comprehensive model evaluation, metrics calculation, and error analysis tools.

## Tools

### Classification Evaluation

- **evaluate_classification_model**: Comprehensive evaluation (accuracy, precision, recall, F1)
- **get_classification_report**: Detailed per-class metrics
- **calculate_confusion_matrix**: Confusion matrix with optional normalization

### Regression Evaluation

- **evaluate_regression_model**: Regression metrics (R², MSE, RMSE, MAE, MAPE)
- **calculate_residuals**: Residual statistics and analysis

### Predictions

- **predict_with_model**: Make predictions on new data
- **compare_model_predictions**: Compare two models side-by-side

### Error Analysis

- **error_analysis**: Identify and analyze prediction errors

## Usage

Add to agent configuration:

```yaml
mcp_server_ids:
  - model_evaluation
```

## Example Tool Calls

```python
# Evaluate classification model
evaluate_classification_model(
    model_path="model.pkl",
    test_data_path="test.csv",
    target_column="target"
)

# Evaluate regression model
evaluate_regression_model(
    model_path="regression_model.pkl",
    test_data_path="test.csv",
    target_column="price"
)

# Get classification report
get_classification_report(
    model_path="model.pkl",
    test_data_path="test.csv",
    target_column="target"
)

# Compare two models
compare_model_predictions(
    model1_path="model1.pkl",
    model2_path="model2.pkl",
    test_data_path="test.csv",
    target_column="target",
    task="classification"
)

# Error analysis
error_analysis(
    model_path="model.pkl",
    test_data_path="test.csv",
    target_column="target",
    task="classification"
)
```

## Metrics Provided

### Classification

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Per-class metrics

### Regression

- R² Score
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Residual statistics
