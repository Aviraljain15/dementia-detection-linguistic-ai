import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report

from feature_extractor import LinguisticFeatureExtractor


def run_xgboost(samples):

    extractor = LinguisticFeatureExtractor()

    # ---------------- FEATURE EXTRACTION ---------------- #
    feature_rows = [extractor.extract(s) for s in samples]
    X = pd.DataFrame(feature_rows).fillna(0)
    y = np.array([s.label for s in samples])
    groups = np.array([s.participant_id for s in samples])

    # ---------------- MODEL ---------------- #
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.2,
        eval_metric='logloss',
        random_state=42,
    )

    # ---------------- CROSS VALIDATION ---------------- #
    gkf = GroupKFold(n_splits=10)

    y_pred = cross_val_predict(model, X, y, cv=gkf, groups=groups)

    print("\n===== XGBOOST RESULTS =====\n")
    print(classification_report(y, y_pred, target_names=['Control', 'Dementia']))

    # ---------------- TRAIN FINAL MODEL ---------------- #
    model.fit(X, y)

    # ---------------- FEATURE IMPORTANCE ---------------- #
    importance = pd.Series(model.feature_importances_, index=X.columns)

    print("\nTop Features:")
    print(importance.sort_values(ascending=False).head(10))
    # RETURN FIXED OUTPUTS
    return model, y_pred, list(X.columns)