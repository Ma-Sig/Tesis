# Data Preparation Tools MCP Server

Provides advanced data transformation, feature engineering, and dataset preparation tools.

## Tools

### Data Cleaning

- **handle_missing_values**: Handle missing values with various strategies (mean, median, mode, drop)
- **remove_duplicates**: Remove duplicate rows from dataset
- **detect_outliers**: Detect outliers using IQR or Z-score methods
- **filter_rows**: Filter rows based on conditions

### Feature Engineering

- **scale_features**: Scale numerical features (StandardScaler or MinMaxScaler)
- **encode_categorical**: Encode categorical variables (Label or One-Hot encoding)
- **create_feature_bins**: Create bins/categories from continuous features

### Dataset Management

- **split_dataset**: Split dataset into training and testing sets

## Usage

Add to agent configuration:

```yaml
mcp_server_ids:
  - data_preparation
```

## Example Tool Calls

```python
# Handle missing values
handle_missing_values(filepath="data.csv", strategy="mean", output_filepath="data_cleaned.csv")

# Scale features
scale_features(filepath="data.csv", columns=["age", "income"], method="standard")

# Encode categorical
encode_categorical(filepath="data.csv", columns=["gender", "city"], method="onehot")

# Split dataset
split_dataset(filepath="data.csv", train_ratio=0.8)
```
