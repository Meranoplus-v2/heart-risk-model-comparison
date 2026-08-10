# Heart Risk Prediction

Binary classification pipeline that trains and compares five models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) to predict `Heart_Risk` from symptom and lifestyle features.

## Dataset

`heart_disease_risk_dataset_earlymed.csv` — binary symptom/lifestyle flags (Chest_Pain, Fatigue, Dizziness, etc.) plus Age and Gender, with a balanced 50/50 target split. Not included in this repo — place it in the project root before running.

## Pipeline

1. Load data, drop exact duplicate rows (prevents the same row from leaking across train/val/test)
2. Stratified split into train / val / test
3. Train five models (Logistic Regression uses a `StandardScaler` pipeline; tree-based models use raw features)
4. Evaluate on val and test sets: Precision, Recall, F1, ROC-AUC, PR-AUC, and a train-vs-eval PR-AUC gap as an overfitting check
5. Report the best model by val PR-AUC
6. Plot feature importance two ways: correlation with the target (`feature_importance_dataset`) and each tree model's own `.feature_importances_` (`feature_importance_models`) — plots are saved as PNGs, not shown interactively

## Run

```bash
pip install -r requirements.txt
python pipeline.py
```

Output plots are written to the project root:
- `dataset_feature_correlation.png`
- `<model_name>_feature_importance.png` for each tree-based model (rf, xgb, catboost, lgbm)

## Note on performance

All models, including plain logistic regression, score 99%+ on Precision/Recall/PR-AUC. This was investigated and is **not data leakage** — checked for duplicate rows across splits, correlated/proxy features, and label purity per feature combination. The result: for this dataset, the target is almost a deterministic function of a handful of symptom flags (combo-level label purity sits at ~0.998–1.0 across nearly every combination).

This investigation was done ad hoc while exploring the data and isn't currently captured as a script in this repo — take the summary above as a description of the process, not a reproducible artifact yet.

This behavior is expected for this dataset but would **not** hold on real clinical data, where symptom-to-diagnosis mapping is much noisier and outcomes overlap even for identical symptom profiles. These results demonstrate pipeline correctness, not a realistic estimate of real-world clinical predictive performance.
