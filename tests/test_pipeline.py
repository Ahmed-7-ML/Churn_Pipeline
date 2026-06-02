import pytest
import pandas as pd
from src.data_validation import validate_data

def test_validation_success():
    valid_data = pd.DataFrame({
            'customer_id': [1, 2],
            'tenure_months': [12, 24],
            'monthly_charges': [50.0, 70.0],
            'total_charges': [600.0, 1680.0]
        })
    
    expected_cols = ['customer_id', 'tenure_months', 'monthly_charges', 'total_charges']
    assert validate_data(valid_data, expected_cols) == True


def test_validation_missing_column():
    broken_data = pd.DataFrame({
        'customer_id': [1, 2],
        'tenure_months': [12, 24]
    })
    expected_cols = ['customer_id', 'tenure_months', 'monthly_charges', 'total_charges']
    assert validate_data(broken_data, expected_cols) == False

def test_validation_negative_values():
    negative_data = pd.DataFrame({
        'customer_id': [1],
        'tenure_months': [-5],  # <-----
        'monthly_charges': [50.0],
        'total_charges': [600.0]
    })
    expected_cols = ['customer_id', 'tenure_months', 'monthly_charges', 'total_charges']
    assert validate_data(negative_data, expected_cols) == False
