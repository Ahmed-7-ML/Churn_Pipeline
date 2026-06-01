# This script will import the logic that we wrote in the previous files, 
# and decorate it with ZenML Decorators (@step and @pipeline).

# Steps:
# Validate Data --> Training ML Model --> Inference

import yaml
import pandas as pd
import logging
from zenml import pipeline, step
from src.data_validation import validate_data, load_config
from src.train import train_model
from src.inference import run_inference


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- 1. First step: Checking daily data ---
@step(enable_cache=False)  # --> # We locked the cash here because the daily data is constantly changing
def validation_step() -> bool:
    config = load_config()
    logging.info("📢 [ZenML Step] Starting Data Validation...")
    
    # Daily Inference Data Check
    df = pd.read_csv(config['data']['inference_data_path'])
    expected_cols = config['validation']['expected_columns']
    
    is_valid = validate_data(df, expected_cols)
    
    return is_valid


# --- 2. Step Two: Training (Optional or Automatic) ---
@step(experiment_tracker="mlflow_tracker")
def training_step(validation_passed: bool):
    if not validation_passed:
        logging.error("❌ [ZenML Step] Validation failed upstream. Skipping Training.")
        return
    
    logging.info("📢 [ZenML Step] Data passed validation. Checking/Running Training...")
    train_model()


# --- 3. Step Three: Inference ---
@step(enable_cache=False)
def inference_step(validation_passed: bool):
    if not validation_passed:
        logging.error("❌ [ZenML Step] Validation failed upstream. Inference Cancelled.")
        return
    
    logging.info("📢 [ZenML Step] Running Batch Inference...")
    run_inference()


# --- 4. Pipeline Assembly ---
@pipeline(enable_cache=False)
def customer_churn_pipeline():
    # Linking the steps together and identifying dependencies
    valid = validation_step()
    
    # The Inference and Train will not work until the validation_step is finished and the result is returned.
    training_step(validation_passed=valid)
    inference_step(validation_passed=valid)

if __name__ == "__main__":
    # Run Pipeline
    customer_churn_pipeline()
