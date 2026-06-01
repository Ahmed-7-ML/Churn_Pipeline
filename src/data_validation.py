# If Failure exists -> Fail Fast

import pandas as pd
import yaml
import logging

# Setting up the logging so we can see what the system is doing step by step
logging.basicConfig(level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path="configs/config.yaml"):
    """Read Config File"""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def validate_data(df: pd.DataFrame, expected_cols: list) -> bool:
    """Inspection of Data and Columns.
    Return True if all is Correct and Valid
    and False otherwise.
    """
    
    # 1/ Schema Check (Check all columns exist)
    current_cols = list(df.columns)
    missing_cols = [col for col in expected_cols if col not in current_cols]
    if missing_cols:
        logging.error(f"❌ The test failed! There are missing columns: {missing_cols}")
        return False
    
    # 2/ Missing Values Check in critical columns
    critical_cols = ['customer_id', 'monthly_charges']
    for col in critical_cols:
        if df[col].isnull().any():
            logging.error(f"❌ Check failed! There are null values ​​in column: {col}")
            return False
    
    # 3/ Data Range Check
    # Ex: The tenure_months (number of subscription months) cannot be negative.
    if (df['tenure_months'] < 0).any():
        logging.error("❌ The test failed! There are illogical negative values ​​in the tenure_months column.")
        return False
    
    # This will run if it passes all checks
    logging.info("✅ Data checked successfully! The data conforms to specifications and is safe to use.")
    return True

if __name__ == "__main__":
    config = load_config()
    expected_cols = config['validation']['expected_columns']
    
    logging.info("The validity of the training data is being checked...")
    df_raw = pd.read_csv(config['data']['raw_data_path'])
    is_valid = validate_data(df_raw, expected_cols)
    print(f"Check result: {is_valid}\n")