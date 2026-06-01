import pandas as pd
from src.data_validation import validate_data, load_config

config = load_config()
expected_cols = config['validation']['expected_columns']

# 1. Read the Valid Data
df = pd.read_csv(config['data']['raw_data_path'])

# 2. We're going to deliberately mess it up: we're going to completely delete the 'monthly_charges' column!
df_broken = df.drop(columns=['monthly_charges'])

print("---The system test data was deliberately corrupted.---")
result = validate_data(df_broken, expected_cols)
print(f"Did the system allow the data to pass through? {result}")