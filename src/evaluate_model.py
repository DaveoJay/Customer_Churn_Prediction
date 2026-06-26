"""
evaluate_model.py

Reusable evaluation and interpretation
functions for trained machine learning models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)


def print_classification_report(
    y_true,
    y_pred
):
    """
    Print the classification report.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like
    """

    print(classification_report(
        y_true,
        y_pred
    ))


def calculate_roc_auc(
    y_true,
    y_prob
):
    """
    Compute ROC-AUC score.

    Parameters
    ----------
    y_true : array-like

    y_prob : array-like

    Returns
    -------
    float
    """

    return roc_auc_score(
        y_true,
        y_prob
    )


def plot_confusion_matrix(
    model,
    X_test,
    y_test
):
    """
    Plot the confusion matrix for a
    trained classification model.

    Parameters
    ----------
    model : estimator

    X_test : DataFrame

    y_test : Series
    """

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()

    plt.title("Confusion Matrix")

    plt.show()


def feature_importance(
    model,
    feature_names
):
    """
    Extract feature importance values
    from a trained Random Forest model.

    Parameters
    ----------
    model : RandomForestClassifier

    feature_names : list

    Returns
    -------
    pd.DataFrame
    """

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance":
            model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    ).reset_index(drop=True)

    return importance


def shap_summary(
    model,
    X
):
    """
    Generate SHAP summary plot.

    Supports both newer and older
    SHAP versions.

    Parameters
    ----------
    model : RandomForestClassifier

    X : DataFrame
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):

        churn_values = shap_values[1]

    else:

        churn_values = shap_values[:, :, 1]

    shap.summary_plot(
        churn_values,
        X
    )