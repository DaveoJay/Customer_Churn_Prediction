"""
business_analysis.py

Business-oriented utilities for interpreting
machine learning predictions in the context of
customer retention.

This module provides functions to:

- Calculate churn probabilities
- Classify customer risk levels
- Identify high-risk customers
- Estimate financial savings
- Summarize business insights
"""

import pandas as pd


def create_prediction_dataframe(
    X: pd.DataFrame,
    model
) -> pd.DataFrame:
    """
    Create a dataframe containing customer
    features, predicted churn probabilities,
    and predicted churn labels.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataset.

    model : trained estimator
        Trained machine learning model.

    Returns
    -------
    pd.DataFrame
        Evaluation dataframe.
    """

    # Copy original feature set
    df_eval = X.copy()

    # Predict churn probability
    df_eval["Churn_Probability"] = model.predict_proba(X)[:, 1]

    # Predict churn class
    df_eval["Predicted_Churn"] = model.predict(X)

    return df_eval


def identify_high_risk_customers(
    df: pd.DataFrame,
    threshold: float = 0.60
) -> pd.DataFrame:
    """
    Identify customers with churn probability
    above a specified threshold.

    Parameters
    ----------
    df : pd.DataFrame

    threshold : float
        Minimum probability required to classify
        a customer as high risk.

    Returns
    -------
    pd.DataFrame
    """

    high_risk = df[
        df["Churn_Probability"] >= threshold
    ]

    return high_risk


def calculate_business_savings(
    high_risk_count: int,
    churn_cost: float = 100,
    retention_cost: float = 20
) -> float:
    """
    Estimate the financial savings achieved by
    retaining high-risk customers.

    Parameters
    ----------
    high_risk_count : int

    churn_cost : float
        Estimated cost of losing one customer.

    retention_cost : float
        Estimated cost of retaining one customer.

    Returns
    -------
    float
        Estimated business savings.
    """

    # Cost if customers are lost
    loss_without_model = (
        high_risk_count
        * churn_cost
    )

    # Cost of proactive retention
    cost_with_model = (
        high_risk_count
        * retention_cost
    )

    # Estimated savings
    savings = (
        loss_without_model
        - cost_with_model
    )

    return savings


def assign_risk_level(
    probability: float
) -> str:
    """
    Assign a customer risk category based
    on predicted churn probability.

    Parameters
    ----------
    probability : float

    Returns
    -------
    str
    """

    if probability < 0.30:

        return "Low Risk"

    elif probability < 0.60:

        return "Medium Risk"

    else:

        return "High Risk"


def add_risk_levels(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add customer risk categories to the
    evaluation dataframe.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df["Risk_Level"] = (
        df["Churn_Probability"]
        .apply(assign_risk_level)
    )

    return df


def risk_summary(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize the number of customers
    within each risk category.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    summary = (
        df["Risk_Level"]
        .value_counts()
        .reset_index()
    )

    summary.columns = [

        "Risk Level",

        "Customer Count"

    ]

    return summary


def business_summary(
    df: pd.DataFrame,
    threshold: float = 0.60,
    churn_cost: float = 100,
    retention_cost: float = 20
) -> dict:
    """
    Generate a complete business summary
    of the churn prediction model.

    Parameters
    ----------
    df : pd.DataFrame

    threshold : float

    churn_cost : float

    retention_cost : float

    Returns
    -------
    dict
    """

    high_risk = identify_high_risk_customers(
        df,
        threshold
    )

    savings = calculate_business_savings(
        len(high_risk),
        churn_cost,
        retention_cost
    )

    summary = {

        "Total Customers":
            len(df),

        "High Risk Customers":
            len(high_risk),

        "Estimated Savings":
            savings

    }

    return summary