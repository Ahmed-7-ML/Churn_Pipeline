"""
This script will run automatically every night (```Batch``` Inference). 
Its functions are:
    1/ To read the new daily data file (daily_inference_data.csv).
    2/ To validate it as well (so that if the new data has a problem, it stops before using the model).
    3/ To load the model we just saved (churn_model.pkl).
    4/ To generate the predictions and save them to a new file (predictions.csv) along with the customer ID so the sales team can identify the intended customer.
"""

# Read Data and Validate it
import yaml
import pandas as pd
from src.data_validation import validate_data, load_config

# To Log Steps
import logging

# To load the saved model
import pickle

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_inference():
    config = load_config()
    
    # 1. Read data for Batch Inferencing
    logging.info("⏳ Loading daily inference data...")
    df = pd.read_csv(config['data']['inference_data_path'])
    
    # 2. Data Validation Gate
    # Without Churn Column
    expected_cols = config['validation']['expected_columns']
    if not validate_data(df, expected_cols):
        logging.error("❌ Inference stopped! Incoming batch data failed verification.")
        return
    
    # 3. Load ML Model
    logging.info("💾 Loading the trained model artifact...")
    try:
        with open(config['model']['model_path'], 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        logging.error(f"❌ Model file not found at {config['model']['model_path']}. Please run training first!")
        return
    
    # 4. Prepare Features
    X = df.drop(columns=['customer_id'])
    
    # 5. Work on Predictions and Probability (the probability that he will cancel the subscription)
    logging.info("🧠 Running model predictions on the batch...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # Churn Percentage
    
    # 6. Final Result
    output_df = pd.DataFrame({
        'customer_id': df['customer_id'],
        'will_churn': predictions,
        'churn_probability': probabilities
    })
    
    # 7. Save the Results
    output_path = config['data']['output_predictions_path']
    output_df.to_csv(output_path, index=False)
    
    logging.info(f"🎯 Inference completed successfully! Results saved to: {output_path}")
    logging.info(f"📊 Preview of predictions:\n\n{output_df.head()}\n")

if __name__ == "__main__":
    run_inference()