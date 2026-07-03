# Feature Engineering MCP Server

Provides feature engineering operations for machine learning datasets through the Model Context Protocol.

## Available Tools

### Data Transformation

- `add_noise(filepath, columns, noise_level)` - Add Gaussian noise to specified columns in a CSV file
- `normalize(filepath, columns)` - Normalize specified columns in a CSV file
- `pca(filepath, columns, n_components)` - Perform Principal Component Analysis on specified columns
- `select_k_best(filepath, target_column, k)` - Select K best features based on ANOVA F-value
- `mutual_info(filepath, target_column, k)` - Select K best features based on mutual information
- `smote(filepath, target_column)` - Apply SMOTE for imbalanced datasets
