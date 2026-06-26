"""
model_utils.py

Reusable utilities for training, evaluating,
comparing, and saving machine learning models
used in the Customer Churn Prediction project.
"""

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(
    model_name: str,
    y_true,
    y_pred,
    y_prob
) -> dict:
    """
    Evaluate a classification model using
    common performance metrics.

    Parameters
    ----------
    model_name : str
        Name of the model.

    y_true : array-like
        True target labels.

    y_pred : array-like
        Predicted labels.

    y_prob : array-like
        Predicted probabilities for the
        positive class.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    results = {

        "Model": model_name,

        "Accuracy":
            accuracy_score(y_true, y_pred),

        "Precision":
            precision_score(y_true, y_pred),

        "Recall":
            recall_score(y_true, y_pred),

        "F1 Score":
            f1_score(y_true, y_pred),

        "ROC-AUC":
            roc_auc_score(y_true, y_prob)

    }

    return results


def compare_models(results: list) -> pd.DataFrame:
    """
    Convert multiple model evaluation
    dictionaries into a comparison table.

    Parameters
    ----------
    results : list

    Returns
    -------
    pd.DataFrame
    """

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(drop=True)

    return comparison


def save_model(model, filepath: str):
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : estimator

    filepath : str
    """

    joblib.dump(model, filepath)


def load_model(filepath: str):
    """
    Load a saved model.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    Trained model
    """

    return joblib.load(filepath)