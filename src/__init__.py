"""
src package

Contains reusable modules used throughout the
Customer Churn Prediction project.
"""

# ============================
# Preprocessing
# ============================

from .preprocessing import preprocess_data

# ============================
# Feature Engineering
# ============================

from .feature_engineering import engineer_features

# ============================
# Model Utilities
# ============================

from .model_utils import (
    evaluate_model,
    compare_models,
    save_model,
    load_model
)

# ============================
# Evaluation
# ============================

from .evaluation import (
    plot_confusion_matrix,
    feature_importance,
    shap_summary
)

# ============================
# Business Analysis
# ============================

from .business_analysis import (
    create_prediction_dataframe,
    identify_high_risk_customers,
    calculate_business_savings,
    add_risk_levels,
    risk_summary,
    business_summary
)