# Heart Risk Prediction

Binary classification pipeline that trains and compares five models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) to predict `Heart_Risk` from symptom and lifestyle features.

## Dataset

`heart_disease_risk_dataset_earlymed.csv` — binary symptom/lifestyle flags (Chest_Pain, Fatigue, Dizziness, etc.) plus Age and Gender, with a balanced 50/50 target split.

## Pipeline

1. Load data, drop exact duplicate rows (prevents the same row from leaking across train/val/test)
2. Stratified split into train / val / test
3. Train five models (Logistic Regression uses a `StandardScaler` pipeline; tree-based models use raw features)
4. Evaluate on val and test sets: Precision, Recall, F1, ROC-AUC, PR-AUC, and a train-vs-eval PR-AUC gap as an overfitting check
5. Report the best model by PR-AUC and plot correlation-based feature importance

## Run

```bash
pip install -r requirements.txt
python full.py
```

## Note on performance

All models, including plain logistic regression, score 99%+ on Precision/Recall/PR-AUC. This was investigated and is **not data leakage** — confirmed by checking for duplicate rows across splits, correlated/proxy features, and label purity per feature combination. The result: for this dataset, the target is almost a deterministic function of a handful of symptom flags (verified via combo-level label purity, which sits at ~0.998–1.0 across nearly every combination).

This behavior is expected for this dataset but would **not** hold on real clinical data, where symptom-to-diagnosis mapping is much noisier and outcomes overlap even for identical symptom profiles. These results demonstrate pipeline correctness, not a realistic estimate of real-world clinical predictive performance.
