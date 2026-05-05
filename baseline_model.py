from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score

def run_baseline(samples):
    texts = [s.clean_text for s in samples]
    labels = [s.label for s in samples]
    groups = [s.participant_id for s in samples]
    tfidf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=3,
            stop_words=None,
        )),
        ('clf', LogisticRegression(
            C=0.5,
            class_weight='balanced',
            max_iter=1000
        ))
    ])
    gkf = GroupKFold(n_splits=10)
    y_pred = cross_val_predict(tfidf_pipeline, texts, labels, cv=gkf, groups=groups)
    y_prob = cross_val_predict(tfidf_pipeline, texts, labels, cv=gkf, groups=groups, method='predict_proba')
    print("\n===== BASELINE MODEL RESULTS =====\n")
    print(classification_report(labels, y_pred, target_names=['Control', 'Dementia']))
    auc = roc_auc_score(labels, y_prob[:, 1])
    print(f"AUC-ROC: {auc:.4f}")
    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "auc": auc,
        "groups": groups}