"""
preprocessing.py

Reusable preprocessing functions for the
Customer Churn Prediction project.

This module contains functions used to clean
and prepare the Telco Customer Churn dataset
before feature engineering.
"""

import pandas as pd


def convert_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the TotalCharges column from object to numeric.

    Invalid values are converted to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer churn dataset.

    Returns
    -------
    pd.DataFrame
        Updated dataframe.
    """

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df


def handle_missing_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace missing TotalCharges values with zero.

    Missing values correspond to customers with
    zero tenure and therefore no accumulated charges.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records from the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.drop_duplicates()

    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary Yes/No columns as 1/0.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    binary_columns = [

        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn"

    ]

    mapping = {

        "Yes": 1,
        "No": 0

    }

    for column in binary_columns:

        if column in df.columns:

            df[column] = df[column].map(mapping)

    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Gender column.

    Male   -> 1
    Female -> 0

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    if "gender" in df.columns:

        df["gender"] = df["gender"].map({

            "Male": 1,
            "Female": 0

        })

    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply one-hot encoding to all remaining
    categorical variables.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    df = pd.get_dummies(

        df,

        columns=categorical_columns,

        drop_first=True,

        dtype=int

    )

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.

    Steps
    -----
    1. Convert TotalCharges to numeric
    2. Handle missing values
    3. Remove duplicate records
    4. Encode binary variables
    5. Encode gender
    6. Apply one-hot encoding

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Fully cleaned dataset.
    """

    # Convert TotalCharges to numeric
    df = convert_total_charges(df)

    # Replace missing TotalCharges values
    df = handle_missing_total_charges(df)

    # Remove duplicate rows
    df = remove_duplicates(df)

    # Encode Yes/No columns
    df = encode_binary_columns(df)

    # Encode Gender
    df = encode_gender(df)

    # One-hot encode remaining categorical columns
    df = one_hot_encode(df)

    return df