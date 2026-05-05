import shap
import pandas as pd


def run_shap_analysis(model, X, feature_names, sample_index=0):
    """
    Run SHAP analysis on trained XGBoost model

    Args:
        model: trained XGBoost model
        X: feature DataFrame
        feature_names: list of feature names
        sample_index: index for individual explanation
    """

    print("\n===== SHAP ANALYSIS =====\n")

    # ---------------- EXPLAINER ---------------- #
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ---------------- GLOBAL IMPORTANCE ---------------- #
    print("Showing global feature importance...")
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type='bar'
    )

    # ---------------- INDIVIDUAL EXPLANATION ---------------- #
    print(f"\nExplaining sample index: {sample_index}")

    shap.waterfall_plot(
        shap.Explanation(
            values = shap_values[1][sample_index] if isinstance(shap_values, list) else shap_values[sample_index],
            base_values=explainer.expected_value,
            data=X.iloc[sample_index],
            feature_names=feature_names
        )
    )

    # ---------------- DEPENDENCE PLOT ---------------- #
    if 'mattr' in feature_names:
        print("\nShowing MATTR dependence plot...")
        shap.dependence_plot(
            'mattr',
            shap_values,
            X,
            feature_names=feature_names
        )