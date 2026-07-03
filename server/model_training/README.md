# Model Training Tools MCP Server

Provides comprehensive machine learning model training, hyperparameter tuning, and model selection tools.

## Tools

### Model Training

- **train_classification_model**: Train classification models (Logistic, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting)
- **train_regression_model**: Train regression models (Linear, Ridge, Lasso, Decision Tree, Random Forest, SVR, KNN, Gradient Boosting)

### Model Evaluation & Selection

- **cross_validate_model**: Perform k-fold cross-validation
- **compare_models**: Compare multiple models on the same dataset
- **get_feature_importance**: Get feature importance from tree-based models

### Hyperparameter Optimization

- **hyperparameter_tuning**: Grid search for optimal hyperparameters

## Usage

Add to agent configuration:

```yaml
mcp_server_ids:
  - model_training
```

## Example Tool Calls

```python
# Train a classification model
train_classification_model(
    filepath="data.csv",
    target_column="target",
    model_type="random_forest",
    model_save_path="model.pkl"
)

# Compare models
compare_models(filepath="data.csv", target_column="target", task="classification")

# Hyperparameter tuning
hyperparameter_tuning(
    filepath="data.csv",
    target_column="target",
    model_type="random_forest"
)

# Get feature importance
get_feature_importance(filepath="data.csv", target_column="target", top_n=10)
```

## Supported Models

### Classification

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN
- Naive Bayes
- Gradient Boosting

### Regression

- Linear Regression
- Ridge
- Lasso
- Decision Tree
- Random Forest
- SVR
- KNN
- Gradient Boosting
