# src/data_load.py

import pandas as pd

def load_data(file_path, file_type='csv'):
    if file_type == 'csv':
        data = pd.read_csv(file_path)
    elif file_type == 'excel':
        data = pd.read_excel(file_path)
    elif file_type == 'json':
        data = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type. Please use 'csv', 'excel', or 'json'.")
    return data

def save_data(data, file_path, file_type='csv'):
    if file_type == 'csv':
        data.to_csv(file_path, index=False)
    elif file_type == 'excel':
        data.to_excel(file_path, index=False)
    elif file_type == 'json':
        data.to_json(file_path)
    else:
        raise ValueError("Unsupported file type. Please use 'csv', 'excel', or 'json'.")
