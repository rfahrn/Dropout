from sklearn.metrics import roc_auc_score, classification_report
import shap

def evaluate_model(model, X, y, set_name="Test"):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    print(f"{set_name} Set AUC: {auc:.4f}")
    print(classification_report(y, y_pred))
    return auc

def explain_model(model, X, feature_names):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names)
    return shap_values