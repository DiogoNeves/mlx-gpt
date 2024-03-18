"""Data utilities."""

import os
import requests


PWD = os.getcwd()
DATA_DIR = os.path.join(PWD, "data")


def fetch_input_data(data_url: str, filename: str) -> str:
    """Gets the input data, caching it for easy access."""
    input_file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(input_file_path):
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)
    
    with open(input_file_path, "r", encoding="utf-8") as f:
        return f.read()
