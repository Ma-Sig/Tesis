# Data Loading and Preprocessing MCP Server

Provides tools for loading CSV files and performing basic data analysis and validation.

## Tools

### File Loading

- **load_csv**: Load CSV file and return metadata (rows, columns, dtypes, sample)
- **read_file_head**: Read first N rows of a CSV file

### Data Analysis

- **get_column_info**: Get detailed statistics for a specific column
- **get_basic_stats**: Get comprehensive statistical summary of the dataset
- **infer_data_types**: Automatically suggest optimal data types for columns

### Data Quality

- **detect_missing_values**: Detect and report missing values
- **detect_duplicates**: Identify duplicate rows
- **validate_csv_structure**: Validate file structure and integrity

## Usage

Add to agent configuration:

```yaml
mcp_server_ids:
  - data_loading
```

## Example Tool Calls

```python
# Load CSV file
load_csv(filepath="my_data.csv")

# Check missing values
detect_missing_values(filepath="my_data.csv")

# Get column details
get_column_info(filepath="my_data.csv", column_name="age")
```
