"""
data_loader.py

Utility functions for loading datasets used throughout
the Customer Churn Prediction project.
"""

import pandas as pd
from pathlib import Path



# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


def load_cleaned_data():
    """
    Load cleaned churn dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    file_path = ( BASE_DIR / "data" / "processed" / "cleaned_churn.csv" )

    return pd.read_csv(file_path)


def load_feature_engineered_data():
    """
    Load feature engineered dataset.

    Returns
    -------
    pd.DataFrame
        Machine learning ready dataset.
    """

    file_path = ( BASE_DIR / "data" / "processed" / "feature_engineered_churn.csv" )

    return pd.read_csv(file_path)