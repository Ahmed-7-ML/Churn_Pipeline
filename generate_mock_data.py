import numpy as np
import pandas as pd

def create_mock_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': range(1001, 1001 + n_samples),
        'tenure_months': np.random.randint(1, 72, size=n_samples),
        'monthly_charges': np.random.uniform(15.0, 120.0, size=n_samples),
        'total_charges': np.random.uniform(15.0, 5000.0, size=n_samples),
        'churn': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Save Training Data
    df.to_csv("data/raw_customer_data.csv", index = False)
    
    # # Saving Inference Data
    # (New daily data that we need to predict - without the churn column)
    inference_df = df.drop(columns=["churn"]).sample(100)
    inference_df.to_csv("data/daily_inference_data.csv", index = False)
    
    print("Successfully Generated Mock Data.")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    create_mock_data()