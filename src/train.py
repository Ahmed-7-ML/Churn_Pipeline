# Experiment Trasking
import mlflow
import mlflow.sklearn

# Data Reading and Validation
# To Pass data through the check gates
import yaml
import pandas as pd
from src.data_validation import validate_data, load_config

# Machine Learning Modeling & Model Saving
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pickle

# To Log Steps
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model():
    config = load_config()
    
    # 1. Load Data
    logging.info("⏳ Data is being loaded for model training...")
    df = pd.read_csv(config['data']['raw_data_path'])
    
    # 2. Validate Data
    # Original data with target column
    expected_cols = config['validation']['expected_columns'] + ['churn']
    
    if not validate_data(df, expected_cols):
        logging.error("❌ The training process was cancelled because the data did not pass the check!")
        return
    
    # 3. Prepare Features and Target
    X = df.drop(columns=['churn', 'customer_id'])
    y = df['churn']
    
    # 4. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size= config['model']['test_size'],
        random_state=config['model']['random_state']
    )
    
    # --- [Starting MLflow] ---
    # We tell MLflow to start recording the current experiment automatically
    # 5. Train the model using the hyperparameters from the YAML file.
    logging.info("🏋️ Random Forest model training is underway...")
    model = RandomForestClassifier(
        n_estimators=config['model']['n_estimators'],
        random_state=config['model']['random_state']
    )
    model.fit(X_train, y_train)

    # 6. Model Evaluation
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    logging.info("📊 --- Evaluation Metrics ---")
    print("\n🔹 Confusion Matrix:")
    print(f"   Predicted:   [No Churn]  [Churn]")
    print(f"   Actual [No Churn]:  {cm[0][0]:<12} {cm[0][1]}")
    print(f"   Actual [Churn]:     {cm[1][0]:<12} {cm[1][1]}\n")
    
    print("🔹 Classification Report:")
    print(cr)
    logging.info("-----------------------------")
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_param("n_estimators", config['model']['n_estimators'])
    mlflow.log_param("test_size", config['model']['test_size'])
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    logging.info(f"📊 MLflow Logged -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    
    # 7. Model Saving
    model_output_path = config['model']['model_path']
    with open(model_output_path, 'wb') as file:
        pickle.dump(model, file)

    mlflow.sklearn.log_model(model, "churn_model_artifacts")
    
    logging.info(f"💾 The model has been successfully saved as a production-ready file in: {model_output_path}")

if __name__ == "__main__":
    train_model()

# python -m src.train
# (We used -m src.train so that Python would see the folder as a Module and be able to import data_validation without problems in the Paths).