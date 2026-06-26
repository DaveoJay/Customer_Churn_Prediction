"""
feature_engineering.py

Reusable feature engineering functions for the
Customer Churn Prediction project.

This module creates additional predictive features
that improve model performance while keeping the
feature engineering process reproducible.
"""

import pandas as pd


def create_new_customer_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary feature identifying customers
    who have just joined the company.

    Customers with tenure less than or equal to
    six months are considered new customers.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset.

    Returns
    -------
    pd.DataFrame
        Updated dataframe.
    """

    # Customers with tenure <= 6 months
    df["NewCustomer"] = (
        df["tenure"] <= 6
    ).astype(int)

    return df


def create_high_monthly_charge_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary feature identifying customers
    paying above the median monthly charge.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    # Compute the dataset median
    median_charge = df["MonthlyCharges"].median()

    # Flag customers paying above the median
    df["HighMonthlyCharges"] = (
        df["MonthlyCharges"] > median_charge
    ).astype(int)

    return df


def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize customers into tenure groups.

    The tenure groups represent different stages
    of the customer lifecycle.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    # Define tenure intervals
    bins = [0, 12, 24, 48, 72]

    # Labels for each interval
    labels = [
        "0-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-72 Months"
    ]

    # Create grouped tenure feature
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


def create_customer_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate customer lifetime value.

    Customer Value is calculated as:

        MonthlyCharges × Tenure

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    # Estimate customer lifetime revenue
    df["CustomerValue"] = (
        df["MonthlyCharges"]
        * df["tenure"]
    )

    return df


def create_average_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the customer's average monthly spend.

    Adding 1 to tenure avoids division by zero
    for brand new customers.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    # Average spend over customer lifetime
    df["AvgMonthlySpend"] = (
        df["TotalCharges"]
        /
        (df["tenure"] + 1)
    )

    return df


def encode_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the tenure groups.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    # Convert tenure categories into dummy variables
    df = pd.get_dummies(
        df,
        columns=["TenureGroup"],
        drop_first=True,
        dtype=int
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.

    Steps
    -----
    1. Create New Customer flag
    2. Create High Monthly Charges flag
    3. Create Tenure Groups
    4. Estimate Customer Value
    5. Estimate Average Monthly Spend
    6. One-hot encode Tenure Groups

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Feature engineered dataset.
    """

    # Identify newly acquired customers
    df = create_new_customer_flag(df)

    # Identify customers with above-average monthly bills
    df = create_high_monthly_charge_flag(df)

    # Group customers based on tenure
    df = create_tenure_groups(df)

    # Estimate customer lifetime value
    df = create_customer_value(df)

    # Estimate average monthly spend
    df = create_average_monthly_spend(df)

    # Encode tenure categories
    df = encode_tenure_groups(df)

    return df