# 🚀 End-to-End Batch Churn Scoring Pipeline with Data Validation & Tracking

An enterprise-grade, production-ready MLOps pipeline built to solve the customer churn forecasting problem for SaaS businesses. 
This project transitions away from unstable Jupyter Notebooks into an automated, structured, and monitored machine learning architecture.

Developed by: **Eng. Ahmed Akram Amer** Status: **Production Ready (Local Stack)**

---

## 🛠️ Architecture & Core Components

The system follows DevOps/MLOps principles to orchestrate the entire machine learning lifecycle through a unified Direct Acyclic Graph (DAG):

1. **Data Validation Gate**: Protects downstream processes by strictly checking schema alignment, feature data types, missing null values, and logical range boundaries.
2. **Orchestration Layer (ZenML)**: Handles step dependencies, operational states, data flow execution, and pipeline caching.
3. **Experiment Tracking & Artifact Registry (MLflow)**: Automatically logs metrics, hyperparameters, tracking graphs, and packages models into immutable production artifacts.

---

## 📁 Repository Structure

```text
churn_pipeline/
│
├── configs/
│   └── config.yaml            # Centralized hyperparams, paths, and validation rules
│
├── data/                      # Data storage layer (Mock customer & prediction files)
│   ├── raw_customer_data.csv
│   ├── daily_inference_data.csv
│   └── predictions.csv
    └── churn_model.pkl
│
├── src/                       # Production Source Code
│   ├── data_validation.py     # Schema and boundary integrity verification
│   ├── train.py               # Model training and MLflow metrics logging
│   ├── inference.py           # Batch scoring and daily churn probability calculator
│   └── pipeline.py            # ZenML DAG Orchestration definition
│
├── requirements.txt           # Explicit python library dependencies
└── README.md                  # Project documentation
```

💻 Tech Stack & Tooling
Core ML: Python, Pandas, Scikit-Learn

Configuration Management: PyYAML

Pipeline Orchestration: ZenML

Experiment Tracking & Model Registry: MLflow (v3.13+)

🚀 How to Run and Operate the System
1. Environment Setup
Clone the repository, create a virtual environment, and install the production dependencies:

Bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
2. ZenML Stack Initialization
Register and set up the customized MLOps backend tracking stack:

Bash
# Initialize ZenML
zenml init

# Install necessary integrations
zenml integration install mlflow sklearn -y

# Register and activate the tracking stack
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -e mlflow_tracker
zenml stack set mlflow_stack

3. Execution (Running the Pipeline)
Set the Windows tracking environment fallback variable, then invoke the pipeline workflow:

PowerShell
# On Windows PowerShell
$env:MLFLOW_ALLOW_FILE_STORE="true"

# Execute the pipeline
python -m src.pipeline
4. Accessing the Monitoring Dashboards
To inspect the Pipeline DAG Execution Graph (ZenML):

Bash
zenml login --local --blocking
Open http://127.0.0.1:8237 in your browser.

To inspect the Experiment Performance Metrics & Registered Models (MLflow):

Bash
mlflow ui --port 5000
Open http://127.0.0.1:5000 and navigate to the Experiments tab.

🎯 Verification (Done When?)
Fail-Fast Verification: Mutating or breaking column constraints inside daily_inference_data.csv instantly halts the execution at the validation gate, blocking bad models or corrupted inference output.

Reproducibility: Identical pipeline configurations in config.yaml guarantee matching model metrics (accuracy: 0.79, f1_score: 0.125) tracked losslessly across historical runs.
