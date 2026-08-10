import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# ── Config ────────────────────────────────────────────────

config = {
    'random_state':   42,
    'test_size':      0.2,
    'val_test_size':  0.5,   # splits leftover test chunk into equal val/test halves
}

# ── Base model params ─────────────────────────────────────

lr_params = {
    'max_iter':     1000,
    'random_state': 42,
    'n_jobs':       -1,
}


# ── Manual/default model hyperparameters (untuned) ────────

rf_params = {
    'n_estimators': 300,
    'random_state': 42,
    'n_jobs':       -1
}

xgb_params = {
    'n_estimators':  500,
    'learning_rate': 0.1,
    'max_depth':     6,
    'random_state':  42,
    'n_jobs':        -1
}

lgbm_params = {
    'n_estimators':  500,
    'learning_rate': 0.05,
    'num_leaves':    31,
    'random_state':  42,
    'n_jobs':        -1,
    'verbose':       -1,
    'objective':     'binary'
}

cat_params = {
    'iterations':    500,
    'learning_rate': 0.1,
    'depth':         6,
    'random_state':  42,
    'verbose':       0
}

# ── Load data ─────────────────────────────────────────────

df = pd.read_csv("heart_disease_risk_dataset_earlymed.csv")

X = df.drop(columns=["Heart_Risk"])
y = df["Heart_Risk"]

# dedup before splitting so no row can leak across train/val/test
print("Before dedup:", X.shape)
combined = pd.concat([X, y], axis=1).drop_duplicates()
X = combined.drop(columns=[y.name])
y = combined[y.name]
print("After dedup:", X.shape)

# train / val / test split (stratified to preserve class balance in each set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=config["random_state"],
    test_size=config["test_size"], stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, random_state=config["random_state"],
    test_size=config["val_test_size"], stratify=y_test
)

# ── Models ────────────────────────────────────────────────

def create_logistic_regression(lr_params):
    # wrapped in a pipeline so scaling only happens for LR, not the tree models
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**lr_params))
    ])

models = {
    "lr":       create_logistic_regression(lr_params),
    "rf":       RandomForestClassifier(**rf_params),
    "xgb":      XGBClassifier(**xgb_params),
    "catboost": CatBoostClassifier(**cat_params),
    "lgbm":     LGBMClassifier(**lgbm_params)
}

def train_models(models, X_train, y_train):
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

# ── Evaluation ────────────────────────────────────────────

def val_eval(trained_models, X_train, y_train, X_val, y_val):
    val_results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_proba_train = model.predict_proba(X_train)[:, 1]

        val_pr_auc = average_precision_score(y_val, y_proba)
        train_pr_auc = average_precision_score(y_train, y_proba_train)

        val_results[name] = {
            "confusion_matrix": confusion_matrix(y_val, y_pred),
            "Precision":        precision_score(y_val, y_pred, zero_division=0),
            "Recall":           recall_score(y_val, y_pred, zero_division=0),
            "F1":               f1_score(y_val, y_pred, zero_division=0),
            "ROC-AUC":          roc_auc_score(y_val, y_proba),
            "PR-AUC":           val_pr_auc,
            "Train PR-AUC":     train_pr_auc,
            "Overfit Gap":      round(train_pr_auc - val_pr_auc, 4)
        }

    val_df = pd.DataFrame(val_results).T.sort_values("PR-AUC", ascending=False)
    print("\n" + "="*50)
    print("VAL RESULTS")
    print("="*50)
    print(val_df)
    return val_df, val_results


def test_eval(trained_models, X_train, y_train, X_test, y_test):
    test_results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_proba_train = model.predict_proba(X_train)[:, 1]

        test_pr_auc = average_precision_score(y_test, y_proba)
        train_pr_auc = average_precision_score(y_train, y_proba_train)

        test_results[name] = {
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "Precision":        precision_score(y_test, y_pred, zero_division=0),
            "Recall":           recall_score(y_test, y_pred, zero_division=0),
            "F1":               f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC":          roc_auc_score(y_test, y_proba),
            "PR-AUC":           test_pr_auc,
            "Train PR-AUC":     train_pr_auc,
            "Overfit Gap":      round(train_pr_auc - test_pr_auc, 4)
        }

    test_df = pd.DataFrame(test_results).T.sort_values("PR-AUC", ascending=False)
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(test_df)
    return test_df, test_results


def find_best_model(df):
    # df must already be sorted so the top row is the best model
    best_name = df.index[0]
    best = df.iloc[0]
    print(f"\nBEST MODEL: {best_name}")
    print(f"   PR-AUC   : {best['PR-AUC']:.4f}")
    print(f"   ROC-AUC  : {best['ROC-AUC']:.4f}")
    print(f"   F1       : {best['F1']:.4f}")
    return best_name


def feature_importance_dataset(X, y, title="Feature Importance"):
    # correlation-based proxy; use feature_importance_models() below for tree models instead
    correlations = pd.concat([X, y], axis=1).corr()[y.name].drop(y.name)
    feature_importance = correlations.abs().sort_values(ascending=False)

    print("\n" + "="*50)
    print("FEATURE IMPORTANCE (Correlation with Target)")
    print("="*50)
    print(feature_importance)

    feature_importance.plot(kind='barh', figsize=(10, 6), title=title)
    plt.xlabel("Correlation Strength")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("dataset_feature_correlation.png")

    return feature_importance


def feature_importance_models(trained_models, X):
    # actual model-derived importances (tree models only; LR pipeline has no .feature_importances_)
    importances = {}
    for name, model in trained_models.items():
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            importances[name] = imp

            print("\n" + "="*50)
            print(f"FEATURE IMPORTANCE — {name}")
            print("="*50)
            print(imp)

            imp.plot(kind='barh', figsize=(10, 6), title=f"Feature Importance — {name}")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{name}_feature_importance.png")   # unique filename per model, otherwise each loop overwrites the last
            plt.close()

    return importances

# ── Run ───────────────────────────────────────────────────

if __name__ == "__main__":
    trained_models = train_models(models, X_train, y_train)
    val_df, val_results = val_eval(trained_models, X_train, y_train, X_val, y_val)
    test_df, test_results = test_eval(trained_models, X_train, y_train, X_test, y_test)
    best_model_name = find_best_model(val_df)
    feature_importance_dataset(X, y)
    feature_importance_models(trained_models, X)
