import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
# optimization tool for hyperparameter tuning

def train_model(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric="auc")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=0)
        y_val_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_val_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="auc")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
    return model